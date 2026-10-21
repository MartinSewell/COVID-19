// The Effectiveness of Lockdowns, Face Masks and Vaccination Programmes Vis-a-Vis Mitigating COVID-19
// Martin Sewell
// martin.sewell@cantab.net
// 22 October 2024
// Reviewed 21 October 2026: explicit units, calendar-year boundaries, cumulative
// totals, validated clustering input and safer file handling. See AUDIT_REPORT.txt.
// Updated 21 October 2026: smaller stack frames, streamed cluster/pair exports,
// and the duplicated date-range test removed. See WINDOWS_BUILD_NOTES.txt.
// Define OWID_SKIP_DTW only when intentionally omitting the final DTW stage.

// INPUT FILES: [all downloaded on 1 August 2024]
// Info on data: [For information only, do not need this.]
// https://covid.ourworldindata.org/data/owid-covid-codebook.csv

// Lockdowns
// https://ourworldindata.org/covid-stringency-index
// owid-covid-data.csv [column 47 (where first col is 0)]
// stringency_index
// https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-data.csv [direct download from here]

// Facemasks
// https://ourworldindata.org/grapher/face-covering-policies-covid
// face-covering-policies-covid.csv
// Entity, Code, Day, facial_coverings
// Afghanistan, AFG, 2020 - 01 - 01, 0
// Zimbabwe,ZWE,2022-01-30,3
// 139137 lines

// Vaccinations
// https://ourworldindata.org/grapher/daily-covid-vaccination-doses-per-capita
// daily-covid-vaccination-doses-per-capita.csv
// Entity,Code,Day,new_vaccinations_smoothed_per_million
// The supplied historical Grapher values are actually per 100 people per day.
// A matched comparison with owid-covid-data.csv validates the scale at runtime.
// Afghanistan, AFG, 2021 - 02 - 23, 0.0033
// Zimbabwe,ZWE,2022-10-09,0.006900000000000001

// Cumulative vaccinations
// https://ourworldindata.org/grapher/covid-vaccination-doses-per-capita
// covid-vaccination-doses-per-capita.csv
// Entity,Code,Day,total_vaccinations_per_hundred
// Afghanistan, AFG, 2021 - 02 - 22, 0
// Zimbabwe,ZWE,2022-10-09,74.89

// COVID-19 deaths
// https://ourworldindata.org/covid-deaths [already downloaded above]
// new_deaths_smoothed_per_million
// owid-covid-data.csv [column 15 (where first col is 0)]
// Used for daily coviddeaths[]; cumcoviddeaths[] uses total_deaths_per_million (column 13).
// iso_code,continent,location,date,total_cases,new_cases.............
// AFG,Asia,Afghanistan,2020-01-03,,0.0,,,0.0,,,0.0,,,0.0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,0.0,54.422,18.6,2.581,1.337,1803.987,,597.029,9.59,,,37.746,0.5,64.83,0.511,41128772.0,,,,
// ZWE,Africa,Zimbabwe,2023-08-23,265716.0,0.0,0.0,5713.0,0.0,0.0,16281.08,0.0,0.0,350.05,0.0,0.0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,42.729,19.6,2.822,1.882,1899.775,21.4,307.846,1.82,1.6,30.7,36.791,1.7,61.49,0.571,16320539.0,,,,

// Excess mortality
// https://ourworldindata.org/grapher/excess-mortality-p-scores-average-baseline-by-age
// p_avg_all_ages
// excess-mortality-p-scores-average-baseline-by-age.csv
// Entity,Code,Day,p_avg_0_14,p_avg_15_64,p_avg_65_74,p_avg_75_84,p_avg_85p,p_avg_all_ages
// Albania,ALB, 2020-01-31,,,,,,-10.65
// Uzbekistan,UZB,2023-06-30,,,,,,20.93

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#ifndef OWID_SKIP_DTW
#include "DTW.hpp"
#endif
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <iomanip>
#include <cstdint>
#include <memory>
#include <utility>

int numdates;
int numentities;

std::vector<int> vaxlow;
std::vector<int> vaxmed;
std::vector<int> vaxhigh;
std::vector <std::string> continent;

#include "boost/algorithm/string.hpp"
#include "boost/date_time/gregorian/gregorian.hpp"
#include "boost/math/interpolators/barycentric_rational.hpp"


std::string tostring(double t) {
    std::string str{ std::to_string(t) };
	int offset{ 1 };
    if (str.find_last_not_of('0') == str.find('.'))
		offset = 0;
    str.erase(str.find_last_not_of('0') + offset, std::string::npos);
    return str;
}

// Mean
long double Mean(std::vector<long double> v)
{
	unsigned long long n = v.size();
	if (v.empty()) return std::numeric_limits<long double>::quiet_NaN();
	long double sum = std::accumulate(v.begin(), v.end(), 0.0L);
	long double mean = sum / (long double)n;
	return mean;
}


int CountryIndex(const std::vector<std::string>& list, const std::string& item)
{
    const auto it = std::find(list.begin(), list.end(), item);
    if (item.empty() || it == list.end())
        throw std::runtime_error("Country code not found: " + item);
    return static_cast<int>(it - list.begin());
}

// Standard CSV quoting, including doubled quotes within a quoted field.
std::vector<std::string> csv_fields(const std::string& line) {
    std::vector<std::string> result;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (quoted && i + 1 < line.size() && line[i+1] == '"') { field += '"'; ++i; }
            else quoted = !quoted;
        } else if (ch == ',' && !quoted) { result.push_back(field); field.clear(); }
        else if (ch != '\r') field += ch;
    }
    if (quoted) throw std::runtime_error("Unclosed quote in CSV row.");
    result.push_back(field);
    return result;
}

// FNV-1a fingerprint prevents stale clusters from being applied to changed matrices.
std::uint64_t file_fingerprint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Could not read " + path);
    std::uint64_t h = 14695981039346656037ULL;
    char ch;
    while (in.get(ch)) { h ^= static_cast<unsigned char>(ch); h *= 1099511628211ULL; }
    return h;
}

/**
 * Compute the p_norm between two 1D c++ vectors.
 *
 * The p_norm is sometimes referred to as the Minkowski norm. Common
 * p_norms include p=2.0 for the euclidean norm, or p=1.0 for the
 * manhattan distance. See also
 * https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm
 *
 * @a 1D vector of m size, where m is the number of dimensions.
 * @b 1D vector of m size (must be the same size as b).
 * @p value of norm to use.
 */
double p_norm(std::vector<double> a, std::vector<double> b, double p) {
	if (a.size() != b.size() || !std::isfinite(p) || p <= 0)
        throw std::invalid_argument("p_norm requires equal-length vectors and a finite positive p.");
	double d = 0;
	for (int i = 0; i < a.size(); i++) {
		d += std::pow(std::abs(a[i] - b[i]), p);
	}
	return std::pow(d, 1.0 / p);
}

std::vector<double> Interpolate(std::vector<double> v) {
    std::vector<double> interpolated;
	interpolated = v;
	std::vector<double> x;
	std::vector<double> y;
	for (unsigned int d = 0; d < v.size(); d++) {
		if (!std::isnan(v[d])) {
			x.push_back(double(d));
			y.push_back(v[d]);
		}
	}
	int first = -1;
	int last = -1;
	bool firstcase = true;
	for (unsigned int d = 0; d < interpolated.size(); d++)
		if (!std::isnan(v[d])) {
			if (firstcase == true) {
				first = d;
				firstcase = false;
			}
			last = d;
		}
	if (x.size() > 1) {
		boost::math::interpolators::barycentric_rational<double> interpolant(x.data(), y.data(), x.size(), 0);
		for (unsigned int d = 0; d < interpolated.size(); d++) {
			if (first <= d && d <= last) {
				interpolated[d] = interpolant(double(d));
			}
		}
	}
	x.clear();
	y.clear();
	return interpolated;
}




// Write one cluster/tertile pair at a time. Large matrices are read by reference.
int doclustering(int c, const std::vector<int>& clustcd,
                 const std::vector<int>& clustem,
                 const std::vector<std::vector<double>>& vaccinations,
                 const std::vector<double>& /*cumvax*/,
                 const std::vector<std::vector<double>>& coviddeaths,
                 const std::vector<std::vector<double>>& excessmortalitypscoreinterpolated) {
    const std::vector<int>* tertiles[] = {&vaxlow, &vaxmed, &vaxhigh};
    const char* tertile_names[] = {"low", "med", "high"};
    const std::vector<int>* labels[] = {&clustcd, &clustem};
    const char* measures[] = {"cd", "em"};
    const std::vector<std::vector<double>>* outcomes[] = {
        &coviddeaths, &excessmortalitypscoreinterpolated};
    std::vector<int> country_order;
    std::vector<int> country_tertile;
    for (int country = 0; country < numentities; ++country) {
        for (int group = 0; group < 3; ++group) {
            if (std::find(tertiles[group]->begin(), tertiles[group]->end(), country)
                    != tertiles[group]->end()) {
                country_order.push_back(country);
                country_tertile.push_back(group);
                break;
            }
        }
    }
    if (clustcd.size() != country_order.size() || clustem.size() != country_order.size())
        throw std::runtime_error("Cluster labels do not match the country order.");
    for (int measure = 0; measure < 2; ++measure) {
        for (int cluster = 0; cluster < 20; ++cluster) {
            for (int group = 0; group < 3; ++group) {
                const std::string stem = std::to_string(c) + "_clust" + measures[measure]
                    + std::to_string(cluster) + "_vax" + tertile_names[group];
                std::ofstream outcome_file(stem + ".txt");
                std::ofstream vaccine_file(stem + "_vx.txt");
                if (!outcome_file || !vaccine_file)
                    throw std::runtime_error("Could not open cluster output: " + stem);
                std::vector<int> members;
                for (std::size_t row = 0; row < country_order.size(); ++row)
                    if (country_tertile[row] == group && (*labels[measure])[row] == cluster)
                        members.push_back(country_order[row]);
                for (int day = 0; day < numdates; ++day) {
                    for (int country : members) {
                        vaccine_file << vaccinations[country][day] << "\t";
                        outcome_file << (*outcomes[measure])[country][day] << "\t";
                    }
                    outcome_file << '\n';
                    vaccine_file << '\n';
                }
                outcome_file.close();
                vaccine_file.close();
                if (!outcome_file || !vaccine_file)
                    throw std::runtime_error("Could not finish cluster output: " + stem);
            }
        }
    }
    return 0;
}

// Stream objects live on the heap; references stay valid as the pool grows.
// The pool closes them on normal return and during exception unwinding.
class FileStreams {
    std::vector<std::unique_ptr<std::ifstream>> inputs_;
    std::vector<std::unique_ptr<std::ofstream>> outputs_;
public:
    std::ifstream& input() {
        inputs_.push_back(std::make_unique<std::ifstream>());
        return *inputs_.back();
    }
    std::ofstream& output() {
        outputs_.push_back(std::make_unique<std::ofstream>());
        return *outputs_.back();
    }
};

void write_country_pair(const std::string& filename,
                        const boost::gregorian::date& firstdate,
                        const std::vector<double>& first1,
                        const std::vector<double>& first2,
                        const std::vector<double>& second1,
                        const std::vector<double>& second2) {
    if (first1.size() != first2.size() || first1.size() != second1.size()
            || first1.size() != second2.size())
        throw std::runtime_error("Mismatched country-pair lengths: " + filename);
    std::ofstream output(filename);
    if (!output) throw std::runtime_error("Could not open " + filename);
    for (std::size_t day = 0; day < first1.size(); ++day) {
        const auto d = firstdate + boost::gregorian::date_duration(static_cast<long>(day));
        output << std::setfill('0') << std::setw(2) << d.day() << "/"
               << std::setw(2) << d.month().as_number() << "/" << d.year()
               << "\t" << first1[day] << "\t" << first2[day]
               << "\t" << second1[day] << "\t" << second2[day] << '\n';
    }
    output.close();
    if (!output) throw std::runtime_error("Could not finish " + filename);
    std::cout << ".";
}

int main(int argc, char* argv[]) try
{
    FileStreams streams;
    bool prepare_only = false, use_clusters = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--prepare-only") prepare_only = true;
        else if (arg == "--use-clusters") use_clusters = true;
        else throw std::runtime_error("Unknown option: " + arg);
    }

	int numprevaxdates; // Calculated from 8 December 2020, not an assumed first date.

	auto start = std::chrono::high_resolution_clock::now();

	using namespace boost::gregorian;

	std::string outputfilename;
	outputfilename = "owid_output.txt";
	auto& outputfile = streams.output();
	outputfile.open(outputfilename);
	if (!outputfile) throw std::runtime_error("Could not open " + std::string(outputfilename));

    const auto require_header = [](const std::string& path,
            const std::vector<std::pair<std::size_t, std::string>>& expected) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Could not open " + path);
        std::string line;
        if (!std::getline(input, line)) throw std::runtime_error("Empty input: " + path);
        std::vector<std::string> columns;
        columns = csv_fields(line);
        for (const auto& col : expected)
            if (col.first >= columns.size() || boost::trim_copy(columns[col.first]) != col.second)
                throw std::runtime_error("Unexpected CSV header in " + path);
        if (!std::getline(input, line) || line.empty())
            throw std::runtime_error("No data rows in " + path);
    };
    require_header("owid-covid-data.csv", {{0,"iso_code"},{3,"date"},
        {13,"total_deaths_per_million"},{15,"new_deaths_smoothed_per_million"},
        {44,"new_vaccinations_smoothed_per_million"},{47,"stringency_index"}});
    require_header("face-covering-policies-covid.csv", {{0,"Entity"},{1,"Code"},{2,"Day"}});
    require_header("covid-vaccination-doses-per-capita.csv", {{3,"total_vaccinations_per_hundred"}});
    require_header("daily-covid-vaccination-doses-per-capita.csv", {{0,"Entity"},{1,"Code"},{2,"Day"}});
    require_header("excess-mortality-p-scores-average-baseline-by-age.csv", {{8,"p_avg_all_ages"}});
    outputfile << "Units: daily vaccinations = doses per 100 people per day; COVID-19 death rate = deaths per million people per day; excess mortality = P-score (% above/below the expected baseline).\n";
    outputfile << "Country-group curves are unweighted means over available countries; reporting coverage may vary by date.\nGroups use tertiles of positive reported cumulative doses per 100 people, cumulative deaths per million people, or cumulative policy index-point days. Zero/unknown totals are excluded, not assumed to be bad data.\n";
    outputfile << "P-score summaries below are unweighted means of reported weekly/monthly periods, not aggregate annual excess-death percentages.\n";
    outputfile << "The post-vaccination summary uses the common date 8 December 2020, not each country's rollout date.\n\n";

	std::string token;

	std::vector <std::string> display;

	// PARSE FILE TO DETERMINE SET OF ENTITIES AND DATES
	std::vector <std::string> fields;
	std::vector <std::string> entity;
	std::vector <std::string> countryonly;
	std::vector <std::string> code;

	std::string str, dummy, previouscountry, dt;
	date firstdate;
	date lastdate;
	date newdate;
	date d;

	int numcountries = 0;

	// AFG,Asia,Afghanistan,2020-01-03,...
	std::string coviddeathsfilename;
	coviddeathsfilename = "owid-covid-data.csv";
	auto& coviddeathsfile = streams.input();
	coviddeathsfile.open(coviddeathsfilename);
	if (!coviddeathsfile) throw std::runtime_error("Could not open " + std::string(coviddeathsfilename));
	if (coviddeathsfile.fail()) {
		std::cout << "Failed to open " << coviddeathsfilename << std::endl;
		std::cout << "Press Enter";
		std::cin.get();
		return 0;
	}
	while (!coviddeathsfile.eof()) {
		getline(coviddeathsfile, str); // first row is header, so ignore
		getline(coviddeathsfile, str); // get first row of data
		fields = csv_fields(str);
		code.push_back(fields.at(0)); // add first country
		continent.push_back(fields.at(1));
		entity.push_back(fields.at(2));
		if (fields.at(1) != "")
			numcountries++;

		previouscountry = fields.at(2);
		firstdate = from_simple_string(fields.at(3));
		lastdate = from_simple_string(fields.at(3));
		while (getline(coviddeathsfile, str)) {
			if (!str.empty()) {
				try {
					fields = csv_fields(str);
					if (!fields.at(0).empty() && (std::find(code.begin(), code.end(), fields.at(0)) == code.end())) { // if country not in list, add to list
						code.push_back(fields.at(0));
						continent.push_back(fields.at(1));
						entity.push_back(fields.at(2));
						if (fields.at(1) != "")
							numcountries++;
					}
					newdate = from_simple_string(fields.at(3));
					if (newdate < firstdate)
						firstdate = newdate;
					if (newdate > lastdate)
						lastdate = newdate;
				}
				catch (const std::exception& e) {
					throw std::runtime_error(std::string("Invalid input row: ") + e.what());
				}
			}
		}
	}
	std::cout << ".";


	coviddeathsfile.clear();
	coviddeathsfile.seekg(0);
	numentities = entity.size();
	days daterange = lastdate - firstdate;
	numdates = daterange.days() + 1;
    numprevaxdates = (date(2020, 12, 8) - firstdate).days();
    if (numprevaxdates <= 0 || numprevaxdates > numdates)
        throw std::runtime_error("The input does not cover the pre-vaccination period ending 7 December 2020.");

	outputfile << "First date: " << firstdate << std::endl;
	outputfile << "Last date: " << lastdate << std::endl;
	outputfile << "Number of dates: " << numdates << std::endl;
	outputfile << "Number of entities: " << numentities << std::endl;
	outputfile << "Number of countries: " << numcountries << std::endl;
	outputfile << std::endl;


	double tv1, tv2;


	// DATES
	std::string dp, mp;
	std::string datesfilename;
	datesfilename = "dates.txt";
	auto& datesfile = streams.output();
	datesfile.open(datesfilename);
	if (!datesfile) throw std::runtime_error("Could not open " + std::string(datesfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		date_duration dd(d);
		date d2 = firstdate + dd;
		if (d2.day() < 10)
			dp = "0";
		else
			dp = "";
		if (d2.month() < 10)
			mp = "0";
		else
			mp = "";
		datesfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year();
		datesfile << std::endl;
	}
	std::cout << ".";


	std::vector<std::vector<double>> coviddeaths(numentities, std::vector<double>(numdates, std::nan("")));
	std::vector<std::vector<double>> vaccinations(numentities, std::vector<double>(numdates, std::nan("")));
    std::vector<std::vector<double>> vaccination_reference(numentities, std::vector<double>(numdates, std::nan("")));

		



	days dd;
	int dateint;






	///////////////
	// LOCKDOWNS //
	///////////////
	std::string lockdownfilename;
	lockdownfilename = "owid-covid-data.csv";
	auto& lockdownfile = streams.input();
	std::vector<std::vector<double>> lockdown(numentities, std::vector<double>(numdates, std::nan("")));

	lockdownfile.open(lockdownfilename);

	if (!lockdownfile) throw std::runtime_error("Could not open " + std::string(lockdownfilename));
	while (!lockdownfile.eof()) {
		getline(lockdownfile, str); // first row is header, so ignore
		while (getline(lockdownfile, str)) {
			if (!str.empty()) {
				try {
					fields = csv_fields(str);
					d = from_simple_string(fields.at(3));
					dd = d - firstdate;
					dateint = dd.days();
					if (!fields.at(0).empty() && !fields.at(47).empty() && (dateint >= 0 && dateint < numdates)) {
						lockdown[CountryIndex(code, fields.at(0))][dateint] = std::stod(fields.at(47));
					}
				}
				catch (const std::exception& e) {
					throw std::runtime_error(std::string("Invalid input row: ") + e.what());
				}
			}
		}
	}
	lockdownfile.clear();
	lockdownfile.seekg(0);
	std::cout << ".";

	//////////////////////////
    // CUMULATIVE LOCKDOWNS //
	//////////////////////////


    std::vector<double> cumlockdowns(numentities, std::nan(""));
    for (int ci = 0; ci < numentities; ++ci) {
        double sum = 0.0;
        bool observed = false;
        for (double value : lockdown[ci]) if (std::isfinite(value)) {
            sum += value;
            observed = true;
        }
        if (observed) cumlockdowns[ci] = sum;
    }
    std::cout << ".";






	///////////
	// MASKS //
	///////////
	std::vector<std::vector<double>> masks(numentities, std::vector<double>(numdates, std::nan("")));

	std::string masksfilename;
	masksfilename = "face-covering-policies-covid.csv";
	auto& masksfile = streams.input();
	masksfile.open(masksfilename);
	if (!masksfile) throw std::runtime_error("Could not open " + std::string(masksfilename));
	while (!masksfile.eof()) {
		getline(masksfile, str); // first row is header, so ignore
		while (getline(masksfile, str)) {
			if (!str.empty()) {
				try {
					fields = csv_fields(str);
					d = from_simple_string(fields.at(2));
					dd = d - firstdate;
					dateint = dd.days();
					if (!fields.at(1).empty() && std::find(code.begin(), code.end(), fields.at(1)) != code.end() && !fields.at(3).empty() && dateint >= 0 && dateint < numdates) {
						masks[CountryIndex(code, fields.at(1))][dateint] = std::stod(fields.at(3));
					}
				}
				catch (const std::exception& e) {
					throw std::runtime_error(std::string("Invalid input row: ") + e.what());
				}
			}
		}
	}
	masksfile.clear();
	masksfile.seekg(0);
	std::cout << ".";

	//////////////////////
	// CUMULATIVE MASKS //
	//////////////////////

    std::vector<double> cummasks(numentities, std::nan(""));
    for (int ci = 0; ci < numentities; ++ci) {
        double sum = 0.0;
        bool observed = false;
        for (double value : masks[ci]) if (std::isfinite(value)) {
            sum += value;
            observed = true;
        }
        if (observed) cummasks[ci] = sum;
    }
    std::cout << ".";







	//////////////////
	// COVID DEATHS //
	//////////////////

	std::vector<double> cumcoviddeaths(numentities, std::nan(""));
    std::vector<int> latest_covid_death_date(numentities, -1);
	// COVID-19 deaths
	// AFG,Asia,Afghanistan,2020-01-03,,0.0,,,0.0
	while (!coviddeathsfile.eof()) {
		getline(coviddeathsfile, str); // first row is header, so ignore
		while (getline(coviddeathsfile, str)) {
			if (!str.empty()) {
				try {
					fields = csv_fields(str);
					d = from_simple_string(fields.at(3));
					dd = d - firstdate;
					dateint = dd.days();
                    if (dateint >= 0 && dateint < numdates && !fields.at(0).empty()) {
                        const int ci = CountryIndex(code, fields.at(0));
                        if (!fields.at(13).empty() && dateint >= latest_covid_death_date[ci]) {
                            cumcoviddeaths[ci] = std::stod(fields.at(13));
                            latest_covid_death_date[ci] = dateint;
                        }
                        if (!fields.at(44).empty()) vaccination_reference[ci][dateint] = std::stod(fields.at(44));
                    }
					if (!fields.at(0).empty() && !fields.at(15).empty() && (dateint >= 0 && dateint < numdates)) {
						coviddeaths[CountryIndex(code, fields.at(0))][dateint] = std::stod(fields.at(15));
						// Cumulative deaths come from the reported cumulative field below, not a sum of smoothed rates.
					}
				}
				catch (const std::exception& e) {
					throw std::runtime_error(std::string("Invalid input row: ") + e.what());
				}
			}
		}
	}
	coviddeathsfile.clear();
	coviddeathsfile.seekg(0);
	std::cout << ".";


	double tcd1, tcd2;
	std::vector<double> tempcoviddeaths;
	std::vector<int> coviddeathslow;
	std::vector<int> coviddeathsmed;
	std::vector<int> coviddeathshigh;
	for (unsigned int i = 0; i < numentities; i++)
		if (cumcoviddeaths[i] > 0 && !continent[i].empty()) // ignore all zero values and non countries
			tempcoviddeaths.push_back(cumcoviddeaths[i]);
	if (tempcoviddeaths.size() < 3) throw std::runtime_error("Too few eligible countries for tertiles: tempcoviddeaths");
	std::sort(tempcoviddeaths.begin(), tempcoviddeaths.end());
	tcd1 = tempcoviddeaths[std::round((tempcoviddeaths.size() / 3) - 1)];
	tcd2 = tempcoviddeaths[std::round((2 * tempcoviddeaths.size() / 3) - 1)];
	for (unsigned int i = 0; i < numentities; i++)
		if (cumcoviddeaths[i] > 0 && !continent[i].empty()) { // ignore non-countries (aggregates) in death groups
			if (cumcoviddeaths[i] <= tcd1)
				coviddeathslow.push_back(i);
			else if (cumcoviddeaths[i] <= tcd2)
				coviddeathsmed.push_back(i);
			else
				coviddeathshigh.push_back(i);
		}

	outputfile << "Countries with low cumulative COVID-19 deaths per million" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million" << std::endl;
	for (unsigned int i = 0; i < coviddeathslow.size(); i++)
		display.push_back(entity[coviddeathslow[i]] + "\t" + tostring(cumcoviddeaths[coviddeathslow[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();

	outputfile << "Countries with medium cumulative COVID-19 deaths per million" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million" << std::endl;
	for (unsigned int i = 0; i < coviddeathsmed.size(); i++)
		display.push_back(entity[coviddeathsmed[i]] + "\t" + tostring(cumcoviddeaths[coviddeathsmed[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();

	outputfile << "Countries with high cumulative COVID-19 deaths per million" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million" << std::endl;
	for (unsigned int i = 0; i < coviddeathshigh.size(); i++)
		display.push_back(entity[coviddeathshigh[i]] + "\t" + tostring(cumcoviddeaths[coviddeathshigh[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();
	std::cout << ".";
	





	//////////////////////
	// EXCESS MORTALITY //
	//////////////////////

	std::vector<double> cumexcess(numentities);
	std::vector<double> meanexcess(numentities);
	std::vector<unsigned int> sumem(numentities);

	std::vector<double> cumexcesspostvax(numentities);
	std::vector<double> meanexcesspostvax(numentities);
	std::vector<unsigned int> sumempostvax(numentities);

	std::vector<double> cumexcess2021(numentities);
	std::vector<double> meanexcess2021(numentities);
	std::vector<unsigned int> sumem2021(numentities);

	std::vector<double> cumexcess2022(numentities);
	std::vector<double> meanexcess2022(numentities);
	std::vector<unsigned int> sumem2022(numentities);

	std::vector<double> cumexcess2023(numentities);
	std::vector<double> meanexcess2023(numentities);
	std::vector<unsigned int> sumem2023(numentities);


	// excess mortality pscore interpolated
	// Entity, Code, Day, p_avg_0_14, p_avg_15_64, p_avg_65_74, p_avg_75_84, p_avg_85p, p_avg_all_ages
	//   0       1    2        3           4            5            6           7           8
	// Albania,ALB,2020-01-31,,,,,,-10.65

	std::vector<std::vector<double>> excessmortalitypscore(numentities, std::vector<double>(numdates, std::nan("")));

	std::vector<double> HRVexcessmortalitypscore_0_14(numdates, std::nan(""));
	std::vector<double> HRVexcessmortalitypscore_15_64(numdates, std::nan(""));
	std::vector<double> HRVexcessmortalitypscore_65_74(numdates, std::nan(""));
	std::vector<double> HRVexcessmortalitypscore_75_84(numdates, std::nan(""));
	std::vector<double> HRVexcessmortalitypscore_85p(numdates, std::nan(""));

	std::vector<double> HUNexcessmortalitypscore_0_14(numdates, std::nan(""));
	std::vector<double> HUNexcessmortalitypscore_15_64(numdates, std::nan(""));
	std::vector<double> HUNexcessmortalitypscore_65_74(numdates, std::nan(""));
	std::vector<double> HUNexcessmortalitypscore_75_84(numdates, std::nan(""));
	std::vector<double> HUNexcessmortalitypscore_85p(numdates, std::nan(""));

	std::vector<double> HRVexcessmortalitypscore_0_14interpolated(numdates, std::nan(""));
	std::vector<double> HRVexcessmortalitypscore_15_64interpolated(numdates, std::nan(""));
	std::vector<double> HRVexcessmortalitypscore_65_74interpolated(numdates, std::nan(""));
	std::vector<double> HRVexcessmortalitypscore_75_84interpolated(numdates, std::nan(""));
	std::vector<double> HRVexcessmortalitypscore_85pinterpolated(numdates, std::nan(""));

	std::vector<double> HUNexcessmortalitypscore_0_14interpolated(numdates, std::nan(""));
	std::vector<double> HUNexcessmortalitypscore_15_64interpolated(numdates, std::nan(""));
	std::vector<double> HUNexcessmortalitypscore_65_74interpolated(numdates, std::nan(""));
	std::vector<double> HUNexcessmortalitypscore_75_84interpolated(numdates, std::nan(""));
	std::vector<double> HUNexcessmortalitypscore_85pinterpolated(numdates, std::nan(""));

    // Entity,Code,Day,p_avg_0_14,p_avg_15_64,p_avg_65_74,p_avg_75_84,p_avg_85p,p_avg_all_ages
    // Croatia,HRV,2020-01-05,14.29,-7.84,-3.74,-28.37,-8.06,-14.47
	std::string excessmortalitypscorefilename;
	excessmortalitypscorefilename = "excess-mortality-p-scores-average-baseline-by-age.csv";
	auto& excessmortalitypscorefile = streams.input();
	excessmortalitypscorefile.open(excessmortalitypscorefilename);
	if (!excessmortalitypscorefile) throw std::runtime_error("Could not open " + std::string(excessmortalitypscorefilename));
	while (!excessmortalitypscorefile.eof()) {
		getline(excessmortalitypscorefile, str); // first row is header, so ignore
		getline(excessmortalitypscorefile, str); // get first row of data
		fields = csv_fields(str);
		d = from_simple_string(fields.at(2));
		dd = d - firstdate;
		dateint = dd.days();
		if (!fields.at(1).empty() && std::find(code.begin(), code.end(), fields.at(1)) != code.end() && !fields.at(8).empty() && dateint >= 0 && dateint < numdates) {
			excessmortalitypscore[CountryIndex(code, fields.at(1))][dateint] = std::stod(fields.at(8));

			if (fields.at(1) == "HRV") {
				HRVexcessmortalitypscore_0_14[dateint] = (fields.at(3).empty() ? std::nan("") : std::stod(fields.at(3)));
				HRVexcessmortalitypscore_15_64[dateint] = (fields.at(4).empty() ? std::nan("") : std::stod(fields.at(4)));
				HRVexcessmortalitypscore_65_74[dateint] = (fields.at(5).empty() ? std::nan("") : std::stod(fields.at(5)));
				HRVexcessmortalitypscore_75_84[dateint] = (fields.at(6).empty() ? std::nan("") : std::stod(fields.at(6)));
				HRVexcessmortalitypscore_85p[dateint] = (fields.at(7).empty() ? std::nan("") : std::stod(fields.at(7)));
			}


			if (fields.at(1) == "HUN") {
				HUNexcessmortalitypscore_0_14[dateint] = (fields.at(3).empty() ? std::nan("") : std::stod(fields.at(3)));
				HUNexcessmortalitypscore_15_64[dateint] = (fields.at(4).empty() ? std::nan("") : std::stod(fields.at(4)));
				HUNexcessmortalitypscore_65_74[dateint] = (fields.at(5).empty() ? std::nan("") : std::stod(fields.at(5)));
				HUNexcessmortalitypscore_75_84[dateint] = (fields.at(6).empty() ? std::nan("") : std::stod(fields.at(6)));
				HUNexcessmortalitypscore_85p[dateint] = (fields.at(7).empty() ? std::nan("") : std::stod(fields.at(7)));
			}

			cumexcess[CountryIndex(code, fields.at(1))] = cumexcess[CountryIndex(code, fields.at(1))] + std::stod(fields.at(8));

			if (d >= date(2020, 12, 8))
				cumexcesspostvax[CountryIndex(code, fields.at(1))] = cumexcesspostvax[CountryIndex(code, fields.at(1))] + std::stod(fields.at(8));

			if (d.year() == 2021)
				cumexcess2021[CountryIndex(code, fields.at(1))] = cumexcess2021[CountryIndex(code, fields.at(1))] + std::stod(fields.at(8));

			if (d.year() == 2022)
				cumexcess2022[CountryIndex(code, fields.at(1))] = cumexcess2022[CountryIndex(code, fields.at(1))] + std::stod(fields.at(8));

			if (d.year() == 2023)
				cumexcess2023[CountryIndex(code, fields.at(1))] = cumexcess2023[CountryIndex(code, fields.at(1))] + std::stod(fields.at(8));

			sumem[CountryIndex(code, fields.at(1))]++;
			if (d >= date(2020, 12, 8))
				sumempostvax[CountryIndex(code, fields.at(1))]++;
			if (d.year() == 2021)
				sumem2021[CountryIndex(code, fields.at(1))]++;
			if (d.year() == 2022)
				sumem2022[CountryIndex(code, fields.at(1))]++;
			if (d.year() == 2023)
				sumem2023[CountryIndex(code, fields.at(1))]++;
		}




		while (getline(excessmortalitypscorefile, str)) {
			if (!str.empty()) {
				try {
					fields = csv_fields(str);
					d = from_simple_string(fields.at(2));
					dd = d - firstdate;
					dateint = dd.days();
					if (!fields.at(1).empty() && !fields.at(8).empty() && !(std::find(code.begin(), code.end(), fields.at(1)) == code.end()) && (dateint >= 0 && dateint < numdates)) {
						excessmortalitypscore[CountryIndex(code, fields.at(1))][dateint] = std::stod(fields.at(8));

						if (fields.at(1) == "HRV") {
							HRVexcessmortalitypscore_0_14[dateint] = (fields.at(3).empty() ? std::nan("") : std::stod(fields.at(3)));
							HRVexcessmortalitypscore_15_64[dateint] = (fields.at(4).empty() ? std::nan("") : std::stod(fields.at(4)));
							HRVexcessmortalitypscore_65_74[dateint] = (fields.at(5).empty() ? std::nan("") : std::stod(fields.at(5)));
							HRVexcessmortalitypscore_75_84[dateint] = (fields.at(6).empty() ? std::nan("") : std::stod(fields.at(6)));
							HRVexcessmortalitypscore_85p[dateint] = (fields.at(7).empty() ? std::nan("") : std::stod(fields.at(7)));
						}

						if (fields.at(1) == "HUN") {
							HUNexcessmortalitypscore_0_14[dateint] = (fields.at(3).empty() ? std::nan("") : std::stod(fields.at(3)));
							HUNexcessmortalitypscore_15_64[dateint] = (fields.at(4).empty() ? std::nan("") : std::stod(fields.at(4)));
							HUNexcessmortalitypscore_65_74[dateint] = (fields.at(5).empty() ? std::nan("") : std::stod(fields.at(5)));
							HUNexcessmortalitypscore_75_84[dateint] = (fields.at(6).empty() ? std::nan("") : std::stod(fields.at(6)));
							HUNexcessmortalitypscore_85p[dateint] = (fields.at(7).empty() ? std::nan("") : std::stod(fields.at(7)));
						}

						cumexcess[CountryIndex(code, fields.at(1))] = cumexcess[CountryIndex(code, fields.at(1))] + std::stod(fields.at(8));

						if (d >= date(2020, 12, 8))
							cumexcesspostvax[CountryIndex(code, fields.at(1))] = cumexcesspostvax[CountryIndex(code, fields.at(1))] + std::stod(fields.at(8));

						if (d.year() == 2021)
							cumexcess2021[CountryIndex(code, fields.at(1))] = cumexcess2021[CountryIndex(code, fields.at(1))] + std::stod(fields.at(8));

						if (d.year() == 2022)
							cumexcess2022[CountryIndex(code, fields.at(1))] = cumexcess2022[CountryIndex(code, fields.at(1))] + std::stod(fields.at(8));

						if (d.year() == 2023)
							cumexcess2023[CountryIndex(code, fields.at(1))] = cumexcess2023[CountryIndex(code, fields.at(1))] + std::stod(fields.at(8));

						sumem[CountryIndex(code, fields.at(1))]++;
						if (d >= date(2020, 12, 8))
							sumempostvax[CountryIndex(code, fields.at(1))]++;
						if (d.year() == 2021)
							sumem2021[CountryIndex(code, fields.at(1))]++;
						if (d.year() == 2022)
							sumem2022[CountryIndex(code, fields.at(1))]++;
						if (d.year() == 2023)
							sumem2023[CountryIndex(code, fields.at(1))]++;
					}
				}
				catch (const std::exception& e) {
					throw std::runtime_error(std::string("Invalid input row: ") + e.what());
				}
			}
		}
	}
	std::cout << ".";
	for (int i = 0; i < cumexcess.size(); i++)
		meanexcess[i] = sumem[i] ? cumexcess[i] / sumem[i] : std::nan("");

	for (int i = 0; i < cumexcesspostvax.size(); i++)
		meanexcesspostvax[i] = sumempostvax[i] ? cumexcesspostvax[i] / sumempostvax[i] : std::nan("");

	for (int i = 0; i < cumexcess2021.size(); i++)
		meanexcess2021[i] = sumem2021[i] ? cumexcess2021[i] / sumem2021[i] : std::nan("");

	for (int i = 0; i < cumexcess2022.size(); i++)
		meanexcess2022[i] = sumem2022[i] ? cumexcess2022[i] / sumem2022[i] : std::nan("");

	for (int i = 0; i < cumexcess2023.size(); i++)
		meanexcess2023[i] = sumem2023[i] ? cumexcess2023[i] / sumem2023[i] : std::nan("");

	excessmortalitypscorefile.clear();
	excessmortalitypscorefile.seekg(0);



	std::vector<std::vector<double>> excessmortalitypscoreinterpolated(numentities, std::vector<double>(numdates, std::nan("")));



	excessmortalitypscoreinterpolated = excessmortalitypscore;
	std::vector<double> x;
	std::vector<double> y;
	for (unsigned int c = 0; c < excessmortalitypscore.size(); c++) {
		for (unsigned int d = 0; d < excessmortalitypscore[c].size(); d++) {
			if (!std::isnan(excessmortalitypscore[c][d])) {
				x.push_back(double(d));
				y.push_back(excessmortalitypscore[c][d]);
			}
		}
		int first = -1;
		int last = -1;
		bool firstcase = true;
		for (unsigned int d = 0; d < excessmortalitypscoreinterpolated[c].size(); d++)
			if (!std::isnan(excessmortalitypscore[c][d])) {
				if (firstcase == true) {
					first = d;
					firstcase = false;
				}
				last = d;
			}
		if (x.size() > 1) {
			boost::math::interpolators::barycentric_rational<double> interpolant(x.data(), y.data(), x.size(), 0); // approximation order, default is 3, but 0 looks best
			//boost::math::interpolators::cardinal_cubic_b_spline<double> spline(y.begin(), y.end(), first, 1);
			for (unsigned int d = 0; d < excessmortalitypscoreinterpolated[c].size(); d++) {
				if (first <= d && d <= last) {
					excessmortalitypscoreinterpolated[c][d] = interpolant(double(d));
					//excessmortalitypscoreinterpolated[c][d] = spline(double(d));
				}
			}
		}
		x.clear();
		y.clear();
	}
	
    // Fit pre-vaccination histories using only observations before 8 December 2020.
    // Barycentric interpolation is global: slicing the full-history interpolant
    // would allow later mortality observations to influence clustering and DTW.
    std::vector<std::vector<double>> excessmortalitypscoreprevax;
    excessmortalitypscoreprevax.reserve(numentities);
    for (const auto& observations : excessmortalitypscore) {
        std::vector<double> before_vaccination(observations.begin(),
                                             observations.begin() + numprevaxdates);
        excessmortalitypscoreprevax.push_back(Interpolate(std::move(before_vaccination)));
    }

	HRVexcessmortalitypscore_0_14interpolated = Interpolate(HRVexcessmortalitypscore_0_14);
    HRVexcessmortalitypscore_15_64interpolated = Interpolate(HRVexcessmortalitypscore_15_64);
    HRVexcessmortalitypscore_65_74interpolated = Interpolate(HRVexcessmortalitypscore_65_74);
    HRVexcessmortalitypscore_75_84interpolated = Interpolate(HRVexcessmortalitypscore_75_84);
    HRVexcessmortalitypscore_85pinterpolated = Interpolate(HRVexcessmortalitypscore_85p);

    HUNexcessmortalitypscore_0_14interpolated = Interpolate(HUNexcessmortalitypscore_0_14);
    HUNexcessmortalitypscore_15_64interpolated = Interpolate(HUNexcessmortalitypscore_15_64);
    HUNexcessmortalitypscore_65_74interpolated = Interpolate(HUNexcessmortalitypscore_65_74);
    HUNexcessmortalitypscore_75_84interpolated = Interpolate(HUNexcessmortalitypscore_75_84);
    HUNexcessmortalitypscore_85pinterpolated = Interpolate(HUNexcessmortalitypscore_85p);
	
	std::cout << ".";


	std::string coviddeathslowcdfilename;
	coviddeathslowcdfilename = "coviddeathslowcd.txt";
	auto& coviddeathslowcdfile = streams.output();
	coviddeathslowcdfile.open(coviddeathslowcdfilename);
	if (!coviddeathslowcdfile) throw std::runtime_error("Could not open " + std::string(coviddeathslowcdfilename));

	std::string coviddeathslowemfilename;
	coviddeathslowemfilename = "coviddeathslowem.txt";
	auto& coviddeathslowemfile = streams.output();
	coviddeathslowemfile.open(coviddeathslowemfilename);
	if (!coviddeathslowemfile) throw std::runtime_error("Could not open " + std::string(coviddeathslowemfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathslow.size(); i++) {
			coviddeathslowcdfile << coviddeaths[coviddeathslow[i]][d] << "\t";
			coviddeathslowemfile << excessmortalitypscoreinterpolated[coviddeathslow[i]][d] << "\t";
		}
		coviddeathslowcdfile << std::endl;
		coviddeathslowemfile << std::endl;
	}
	std::string coviddeathsmedcdfilename;
	coviddeathsmedcdfilename = "coviddeathsmedcd.txt";
	auto& coviddeathsmedcdfile = streams.output();
	coviddeathsmedcdfile.open(coviddeathsmedcdfilename);
	if (!coviddeathsmedcdfile) throw std::runtime_error("Could not open " + std::string(coviddeathsmedcdfilename));
	std::string coviddeathsmedemfilename;
	coviddeathsmedemfilename = "coviddeathsmedem.txt";
	auto& coviddeathsmedemfile = streams.output();
	coviddeathsmedemfile.open(coviddeathsmedemfilename);
	if (!coviddeathsmedemfile) throw std::runtime_error("Could not open " + std::string(coviddeathsmedemfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathsmed.size(); i++) {
			coviddeathsmedcdfile << coviddeaths[coviddeathsmed[i]][d] << "\t";
			coviddeathsmedemfile << excessmortalitypscoreinterpolated[coviddeathsmed[i]][d] << "\t";
		}
		coviddeathsmedcdfile << std::endl;
		coviddeathsmedemfile << std::endl;
	}
	std::string coviddeathshighcdfilename;
	coviddeathshighcdfilename = "coviddeathshighcd.txt";
	auto& coviddeathshighcdfile = streams.output();
	coviddeathshighcdfile.open(coviddeathshighcdfilename);
	if (!coviddeathshighcdfile) throw std::runtime_error("Could not open " + std::string(coviddeathshighcdfilename));
	std::string coviddeathshighemfilename;
	coviddeathshighemfilename = "coviddeathshighem.txt";
	auto& coviddeathshighemfile = streams.output();
	coviddeathshighemfile.open(coviddeathshighemfilename);
	if (!coviddeathshighemfile) throw std::runtime_error("Could not open " + std::string(coviddeathshighemfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathshigh.size(); i++) {
			coviddeathshighcdfile << coviddeaths[coviddeathshigh[i]][d] << "\t";
			coviddeathshighemfile << excessmortalitypscoreinterpolated[coviddeathshigh[i]][d] << "\t";
		}
		coviddeathshighcdfile << std::endl;
		coviddeathshighemfile << std::endl;
	}
	std::cout << ".";





	// CUMULATIVE VACCINATIONS
	// Generate a list of countries by total cumulative number of vaccinations
	std::vector<double> cumvax(numentities, std::nan(""));
	std::vector<double> cumvax2020(numentities, std::nan(""));
	std::vector<double> cumvax20202021(numentities, std::nan(""));
	std::vector<double> cumvax20202022(numentities, std::nan(""));
    std::vector<int> latest_vax(numentities, std::numeric_limits<int>::min());
    std::vector<int> latest_vax2020 = latest_vax, latest_vax2021 = latest_vax, latest_vax2022 = latest_vax;
    const auto update_latest = [](std::vector<double>& totals, std::vector<int>& latest,
                                  int ci, int date_index, double value) {
        if (date_index >= latest[ci]) { totals[ci] = value; latest[ci] = date_index; }
    };
	// covid-vaccination-doses-per-capita.csv
	// Afghanistan,AFG,2021-02-22,0
	std::string cumvaxfilename;
	cumvaxfilename = "covid-vaccination-doses-per-capita.csv";
	auto& cumvaxfile = streams.input();
	cumvaxfile.open(cumvaxfilename);
	if (!cumvaxfile) throw std::runtime_error("Could not open " + std::string(cumvaxfilename));
	while (!cumvaxfile.eof()) {
		getline(cumvaxfile, str); // first row is header, so ignore
		getline(cumvaxfile, str); // get first row of data
		fields = csv_fields(str);
		d = from_simple_string(fields.at(2));
		dd = d - firstdate;
		dateint = dd.days();
		if (!fields.at(1).empty() && std::find(code.begin(), code.end(), fields.at(1)) != code.end() && !fields.at(3).empty()) {
			update_latest(cumvax, latest_vax, CountryIndex(code, fields.at(1)), dateint, std::stod(fields.at(3)));
                        if (d <= date(2020, 12, 31))
                            update_latest(cumvax2020, latest_vax2020, CountryIndex(code, fields.at(1)), dateint, std::stod(fields.at(3)));
			if (d <= date(2021, 12, 31))
				update_latest(cumvax20202021, latest_vax2021, CountryIndex(code, fields.at(1)), dateint, std::stod(fields.at(3)));
			if (d <= date(2022, 12, 31))
				update_latest(cumvax20202022, latest_vax2022, CountryIndex(code, fields.at(1)), dateint, std::stod(fields.at(3)));
		}
		while (getline(cumvaxfile, str)) {
			if (!str.empty()) {
				try {
					fields = csv_fields(str);
					d = from_simple_string(fields.at(2));
					dd = d - firstdate;
					dateint = dd.days();
					if (!fields.at(1).empty() && std::find(code.begin(), code.end(), fields.at(1)) != code.end() && !fields.at(3).empty()) {
						update_latest(cumvax, latest_vax, CountryIndex(code, fields.at(1)), dateint, std::stod(fields.at(3)));
                        if (d <= date(2020, 12, 31))
                            update_latest(cumvax2020, latest_vax2020, CountryIndex(code, fields.at(1)), dateint, std::stod(fields.at(3)));
						if (d <= date(2021, 12, 31))
							update_latest(cumvax20202021, latest_vax2021, CountryIndex(code, fields.at(1)), dateint, std::stod(fields.at(3)));
						if (d <= date(2022, 12, 31))
							update_latest(cumvax20202022, latest_vax2022, CountryIndex(code, fields.at(1)), dateint, std::stod(fields.at(3)));
					}
				}
				catch (const std::exception& e) {
					throw std::runtime_error(std::string("Invalid input row: ") + e.what());
				}
			}
		}
	}
	cumvaxfile.clear();
	cumvaxfile.seekg(0);
	std::cout << ".";






	// EXCESS MORTALITY FOR CROATIA AND HUNGARY



	// p_avg_0_14,p_avg_15_64,p_avg_65_74,p_avg_75_84,p_avg_85p

	std::string excessstring;
	outputfile << "Continent" << "\t" << "Entity" << "\t" << "Code" << "\t" << "Cumulative lockdown stringency (index-point days)" << "\t" << "Cumulative face-covering policy (index-point days)" << "\t" << "Total vaccine doses per 100 people" << "\t" << "Total COVID-19 deaths per million" << "\t" << "Mean excess mortality P-scores (%)" << std::endl;
	for (unsigned int i = 0; i < numentities; i++) {
		if (!std::isnan(meanexcess[i]))
			excessstring = tostring(meanexcess[i]);
		else
			excessstring = "nan";
		display.push_back(continent [i] + "\t" + entity[i] + "\t" + code[i] + "\t" + tostring(cumlockdowns[i]) + "\t" + tostring(cummasks[i]) + "\t" + tostring(cumvax[i]) + "\t" + tostring(cumcoviddeaths[i]) + "\t" + excessstring);
	}
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();


	std::string excesspostvaxstring;
	std::string excess2021string;
	std::string excess2022string;
	std::string excess2023string;
	outputfile << "Continent" << "\t" << "Entity" << "\t" << "Code" << "\t" << "Cumulative vaccine doses per 100 people, latest reported by 31 December 2020" << "\t" << "Cumulative vaccine doses per 100 people, latest reported by 31 December 2021" << "\t" << "Cumulative vaccine doses per 100 people, latest reported by 31 December 2022" << "\t" << "Total vaccine doses per 100 people" << "\t" << "Mean post vaccination excess mortality P-scores (%)" << "\t" << "Mean 2021 excess mortality P-scores (%)" << "\t" << "Mean 2022 excess mortality P-scores (%)" << "\t" << "Mean 2023 excess mortality P-scores (%)" << "\t" << "Mean excess mortality P-scores (%)" << std::endl;
	for (unsigned int i = 0; i < numentities; i++) {

		if (!std::isnan(meanexcesspostvax[i]))
			excesspostvaxstring = tostring(meanexcesspostvax[i]);
		else
			excesspostvaxstring = "nan";

		if (!std::isnan(meanexcess2021[i]))
			excess2021string = tostring(meanexcess2021[i]);
		else
			excess2021string = "nan";

		if (!std::isnan(meanexcess2022[i]))
			excess2022string = tostring(meanexcess2022[i]);
		else
			excess2022string = "nan";

		if (!std::isnan(meanexcess2023[i]))
			excess2023string = tostring(meanexcess2023[i]);
		else
			excess2023string = "nan";

		if (!std::isnan(meanexcess[i]))
			excessstring = tostring(meanexcess[i]);
		else
			excessstring = "nan";

		display.push_back(continent[i] + "\t" + entity[i] + "\t" + code[i] + "\t" + tostring(cumvax2020[i]) + "\t" + tostring(cumvax20202021[i]) + "\t" + tostring(cumvax20202022[i]) + "\t" + tostring(cumvax[i]) + "\t" + excesspostvaxstring + "\t" + excess2021string + "\t" + excess2022string + "\t" + excess2023string + "\t" + excessstring);
	}
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();
	std::cout << ".";


	outputfile << std::flush;






	//////////////////
	// VACCINATIONS //
	//////////////////


	int matches_per_hundred = 0, matches_per_million = 0;
	std::string vaxfilename;
	vaxfilename = "daily-covid-vaccination-doses-per-capita.csv";
	auto& vaxfile = streams.input();
	vaxfile.open(vaxfilename);
	if (!vaxfile) throw std::runtime_error("Could not open " + std::string(vaxfilename));

	// Afghanistan,AFG,2021-02-22,0
	while (!vaxfile.eof()) {
		getline(vaxfile, str); // first row is header, so ignore
		while (getline(vaxfile, str)) {
			if (!str.empty()) {
				try {
					fields = csv_fields(str);
					d = from_simple_string(fields.at(2));
					dd = d - firstdate;
					dateint = dd.days();
					if (!fields.at(1).empty() && std::find(code.begin(), code.end(), fields.at(1)) != code.end() && !fields.at(3).empty() && dateint >= 0 && dateint < numdates) {
						vaccinations[CountryIndex(code, fields.at(1))][dateint] = std::stod(fields.at(3));
                        const double ref = vaccination_reference[CountryIndex(code, fields.at(1))][dateint];
                        const double val = std::stod(fields.at(3));
                        if (std::isfinite(ref) && ref > 0 && val > 0) {
                            const double ratio = ref / val;
                            if (std::abs(ratio / 10000.0 - 1.0) < 1e-6) ++matches_per_hundred;
                            else if (std::abs(ratio - 1.0) < 1e-6) ++matches_per_million;
                            else throw std::runtime_error("Daily vaccination scale disagrees with OWID reference.");
                        }
					}
				}
				catch (const std::exception& e) {
					throw std::runtime_error(std::string("Invalid input row: ") + e.what());
				}
			}
		}
	}

	vaxfile.clear();
	vaxfile.seekg(0);
	std::cout << ".";




	    if ((matches_per_hundred == 0 && matches_per_million == 0) ||
            (matches_per_hundred > 0 && matches_per_million > 0))
        throw std::runtime_error("Cannot establish a consistent daily vaccination unit.");
    if (matches_per_million > 0)
        for (auto& row : vaccinations) for (auto& value : row) value /= 10000.0;
    outputfile << "Daily vaccination input verified against "
               << matches_per_hundred + matches_per_million
               << " positive OWID matches; output unit: doses per 100 people per day.\n\n";

std::vector<double> tempvax;

	outputfile << "Countries for clustering:" << std::endl;
	for (unsigned int i = 0; i < numentities; i++)
		if (cumvax[i] > 0 && !continent[i].empty()) { // Eligibility rule: positive reported cumulative doses and a known continent.
			outputfile << entity[i] << '\t';
			tempvax.push_back(cumvax[i]);
		}
	outputfile << std::endl;
	outputfile << std::endl;

	if (tempvax.size() < 3) throw std::runtime_error("Too few eligible countries for tertiles: tempvax");
	std::sort(tempvax.begin(), tempvax.end());
	tv1 = tempvax[std::round((tempvax.size() / 3) - 1)];
	tv2 = tempvax[std::round((2 * tempvax.size() / 3) - 1)];




	for (unsigned int i = 0; i < numentities; i++)
		if (cumvax[i] > 0 && !continent[i].empty()) {
			if (cumvax[i] <= tv1)
				vaxlow.push_back(i);
			else if (cumvax[i] <= tv2)
				vaxmed.push_back(i);
			else
				vaxhigh.push_back(i);
		}

	std::cout << ".";







	outputfile << "Countries with low cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < vaxlow.size(); i++)
		display.push_back(entity[vaxlow[i]] + "\t" + tostring(cumvax[vaxlow[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();


	outputfile << "Countries with medium cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < vaxmed.size(); i++)
		display.push_back(entity[vaxmed[i]] + "\t" + tostring(cumvax[vaxmed[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();

	outputfile << "Countries with high cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < vaxhigh.size(); i++)
		display.push_back(entity[vaxhigh[i]] + "\t" + tostring(cumvax[vaxhigh[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();



	std::string vaxlowvxfilename;
	vaxlowvxfilename = "vaxlowvx.txt";
	auto& vaxlowvxfile = streams.output();
	vaxlowvxfile.open(vaxlowvxfilename);
	if (!vaxlowvxfile) throw std::runtime_error("Could not open " + std::string(vaxlowvxfilename));

	std::string vaxlowcdfilename;
	vaxlowcdfilename = "vaxlowcd.txt";
	auto& vaxlowcdfile = streams.output();
	vaxlowcdfile.open(vaxlowcdfilename);
	if (!vaxlowcdfile) throw std::runtime_error("Could not open " + std::string(vaxlowcdfilename));

	std::string vaxlowemfilename;
	vaxlowemfilename = "vaxlowem.txt";
	auto& vaxlowemfile = streams.output();
	vaxlowemfile.open(vaxlowemfilename);
	if (!vaxlowemfile) throw std::runtime_error("Could not open " + std::string(vaxlowemfilename));



	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlow.size(); i++) {
			vaxlowvxfile << vaccinations[vaxlow[i]][d] << "\t";
			vaxlowcdfile << coviddeaths[vaxlow[i]][d] << "\t";
			vaxlowemfile << excessmortalitypscoreinterpolated[vaxlow[i]][d] << "\t";
		}
		vaxlowvxfile << std::endl;
		vaxlowcdfile << std::endl;
		vaxlowemfile << std::endl;
	}
	std::cout << ".";

	std::string vaxmedvxfilename;
	vaxmedvxfilename = "vaxmedvx.txt";
	auto& vaxmedvxfile = streams.output();
	vaxmedvxfile.open(vaxmedvxfilename);
	if (!vaxmedvxfile) throw std::runtime_error("Could not open " + std::string(vaxmedvxfilename));

	std::string vaxmedcdfilename;
	vaxmedcdfilename = "vaxmedcd.txt";
	auto& vaxmedcdfile = streams.output();
	vaxmedcdfile.open(vaxmedcdfilename);
	if (!vaxmedcdfile) throw std::runtime_error("Could not open " + std::string(vaxmedcdfilename));

	std::string vaxmedemfilename;
	vaxmedemfilename = "vaxmedem.txt";
	auto& vaxmedemfile = streams.output();
	vaxmedemfile.open(vaxmedemfilename);
	if (!vaxmedemfile) throw std::runtime_error("Could not open " + std::string(vaxmedemfilename));

	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmed.size(); i++) {
			vaxmedvxfile << vaccinations[vaxmed[i]][d] << "\t";
			vaxmedcdfile << coviddeaths[vaxmed[i]][d] << "\t";
			vaxmedemfile << excessmortalitypscoreinterpolated[vaxmed[i]][d] << "\t";
		}
		vaxmedvxfile << std::endl;
		vaxmedcdfile << std::endl;
		vaxmedemfile << std::endl;
	}
	std::string vaxhighvxfilename;
	vaxhighvxfilename = "vaxhighvx.txt";
	auto& vaxhighvxfile = streams.output();
	vaxhighvxfile.open(vaxhighvxfilename);
	if (!vaxhighvxfile) throw std::runtime_error("Could not open " + std::string(vaxhighvxfilename));

	std::string vaxhighcdfilename;
	vaxhighcdfilename = "vaxhighcd.txt";
	auto& vaxhighcdfile = streams.output();
	vaxhighcdfile.open(vaxhighcdfilename);
	if (!vaxhighcdfile) throw std::runtime_error("Could not open " + std::string(vaxhighcdfilename));

	std::string vaxhighemfilename;
	vaxhighemfilename = "vaxhighem.txt";
	auto& vaxhighemfile = streams.output();
	vaxhighemfile.open(vaxhighemfilename);
	if (!vaxhighemfile) throw std::runtime_error("Could not open " + std::string(vaxhighemfilename));

	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhigh.size(); i++) {
			vaxhighvxfile << vaccinations[vaxhigh[i]][d] << "\t";
			vaxhighcdfile << coviddeaths[vaxhigh[i]][d] << "\t";
			vaxhighemfile << excessmortalitypscoreinterpolated[vaxhigh[i]][d] << "\t";
		}
		vaxhighvxfile << std::endl;
		vaxhighcdfile << std::endl;
		vaxhighemfile << std::endl;
	}
	std::cout << ".";





	// OUTPUT FILES FOR PYTHON
	std::string clusteringcdfilename;
	clusteringcdfilename = "clusteringcd.txt";
	auto& clusteringcdfile = streams.output();
	clusteringcdfile.open(clusteringcdfilename);
	if (!clusteringcdfile) throw std::runtime_error("Could not open " + std::string(clusteringcdfilename));
	for (unsigned int i = 0; i < numentities; i++)
		if (cumvax[i] > 0 && !continent[i].empty()) {
			for (int d = 0; d < numprevaxdates; d++)
				clusteringcdfile << coviddeaths[i][d] << "\t";
			clusteringcdfile << "\n";
		}
	clusteringcdfile.flush();
	clusteringcdfile.close();

	std::string clusteringemfilename;
	clusteringemfilename = "clusteringem.txt";
	auto& clusteringemfile = streams.output();
	clusteringemfile.open(clusteringemfilename);
	if (!clusteringemfile) throw std::runtime_error("Could not open " + std::string(clusteringemfilename));
	for (unsigned int i = 0; i < numentities; i++)
		if (cumvax[i] > 0 && !continent[i].empty()) {
			for (int d = 0; d < numprevaxdates; d++)
				clusteringemfile << excessmortalitypscoreprevax[i][d] << "\t";
	        clusteringemfile << "\n";
		}
	clusteringemfile.flush();
	clusteringemfile.close();

	std::cout << std::endl;
    std::ofstream country_order("clustering_countries.txt");
    if (!country_order) throw std::runtime_error("Could not write clustering_countries.txt");
    for (int i = 0; i < numentities; ++i)
        if (cumvax[i] > 0 && !continent[i].empty()) country_order << code[i] << '\n';
    country_order.close();
    if (prepare_only) { std::cout << "Clustering inputs prepared. Run clustering.py, then rerun with --use-clusters.\n"; return 0; }
	std::cout << "1) Keep clustering.py with clusteringcd.txt, clusteringem.txt and clustering_countries.txt." << std::endl;
	std::cout << "2) Run: py clustering.py" << std::endl;
	std::cout << "3) Keep clusters.txt and clusters.inputs.fnv in this program's working directory." << std::endl;
	if (!use_clusters) { std::cout << "4) Press Enter after clustering.py has finished. " << std::flush; std::cin.get(); }
	


	int mc = 19;
	int c;

	// std::vector<std::vector<double>> coviddeaths(numentities, std::vector<double>(numdates, std::nan("")));
	std::vector<std::vector<int>> clustcd(mc);
	std::vector<std::vector<int>> clustem(mc);

	std::string clustersfilename;
	clustersfilename = "clusters.txt";
	auto& clustersfile = streams.input();
	clustersfile.open(clustersfilename);
	if (!clustersfile) throw std::runtime_error("Could not open " + std::string(clustersfilename));
	std::string line;

	for (int i = 0; i < mc; i++) { // reads in 19 lines
		getline(clustersfile, line);
		std::istringstream iss1(line);
		while (iss1 >> c)
			clustcd[i].push_back(c);
	}
	for (int i = 0; i < mc; i++) { // reads in 19 lines
		getline(clustersfile, line);
		std::istringstream iss2(line);
		while (iss2 >> c)
			clustem[i].push_back(c);
	}
	std::cout << ".";



    const std::size_t expected_countries = vaxlow.size() + vaxmed.size() + vaxhigh.size();
    for (int row = 0; row < mc; ++row) {
        for (const auto* labels : {&clustcd[row], &clustem[row]}) {
            if (labels->size() != expected_countries)
                throw std::runtime_error("clusters.txt has the wrong number of countries; rerun clustering.py.");
            for (int label : *labels)
                if (label < -1 || label >= row + 2)
                    throw std::runtime_error("Invalid cluster label in clusters.txt.");
        }
    }
    std::ifstream fingerprints("clusters.inputs.fnv");
    std::uint64_t cd_hash, em_hash, country_hash;
    if (!(fingerprints >> cd_hash >> em_hash >> country_hash) ||
        cd_hash != file_fingerprint("clusteringcd.txt") ||
        em_hash != file_fingerprint("clusteringem.txt") ||
        country_hash != file_fingerprint("clustering_countries.txt"))
        throw std::runtime_error("Cluster inputs changed or fingerprints are missing; rerun the reviewed clustering.py.");

	// clustcd[c] and clustem[c] only include clustering countries
	// For example, clustcd3 is a list of clusters, when there are three clusters (0, 1, 2)

	for (unsigned int c = 0; c < mc; c++) {
		doclustering(c, clustcd[c], clustem[c], vaccinations, cumvax, coviddeaths, excessmortalitypscoreinterpolated);
	}





	// LOCKDOWNS
	std::vector<double> templockdown;
	std::vector<int> lockdownlow;
	std::vector<int> lockdownmed;
	std::vector<int> lockdownhigh;
	double tl1, tl2;
	for (unsigned int i = 0; i < numentities; i++)
		if (cumlockdowns[i] > 0 && !continent[i].empty()) // ignore all zero values and non countries
			templockdown.push_back(cumlockdowns[i]);
	if (templockdown.size() < 3) throw std::runtime_error("Too few eligible countries for tertiles: templockdown");
	std::sort(templockdown.begin(), templockdown.end());
	tl1 = templockdown[std::round((templockdown.size() / 3) - 1)];
	tl2 = templockdown[std::round((2 * templockdown.size() / 3) - 1)];
	for (unsigned int i = 0; i < numentities; i++)
		if (cumlockdowns[i] > 0 && !continent[i].empty()) {
			if (cumlockdowns[i] <= tl1)
				lockdownlow.push_back(i);
			else if (cumlockdowns[i] <= tl2)
				lockdownmed.push_back(i);
			else
				lockdownhigh.push_back(i);
		}


	outputfile << "Countries with low cumulative stringency" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency (index-point days)" << std::endl;
	for (unsigned int i = 0; i < lockdownlow.size(); i++)
		display.push_back(entity[lockdownlow[i]] + "\t" + tostring(cumlockdowns[lockdownlow[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();

	outputfile << "Countries with medium cumulative stringency" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency (index-point days)" << std::endl;
	for (unsigned int i = 0; i < lockdownmed.size(); i++)
		display.push_back(entity[lockdownmed[i]] + "\t" + tostring(cumlockdowns[lockdownmed[i]]));
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();

	outputfile << "Countries with high cumulative stringency" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency (index-point days)" << std::endl;
	for (unsigned int i = 0; i < lockdownhigh.size(); i++)
		display.push_back(entity[lockdownhigh[i]] + "\t" + tostring(cumlockdowns[lockdownhigh[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();

	std::string lockdownlowldfilename;
	lockdownlowldfilename = "lockdownlowld.txt";
	auto& lockdownlowldfile = streams.output();
	lockdownlowldfile.open(lockdownlowldfilename);
	if (!lockdownlowldfile) throw std::runtime_error("Could not open " + std::string(lockdownlowldfilename));
	std::string lockdownlowemfilename;
	lockdownlowemfilename = "lockdownlowem.txt";
	auto& lockdownlowemfile = streams.output();
	lockdownlowemfile.open(lockdownlowemfilename);
	if (!lockdownlowemfile) throw std::runtime_error("Could not open " + std::string(lockdownlowemfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < lockdownlow.size(); i++) {
			lockdownlowldfile << lockdown[lockdownlow[i]][d] << "\t";
			lockdownlowemfile << excessmortalitypscoreinterpolated[lockdownlow[i]][d] << "\t";
		}
		lockdownlowldfile << std::endl;
		lockdownlowemfile << std::endl;
	}
	std::string lockdownmedldfilename;
	lockdownmedldfilename = "lockdownmedld.txt";
	auto& lockdownmedldfile = streams.output();
	lockdownmedldfile.open(lockdownmedldfilename);
	if (!lockdownmedldfile) throw std::runtime_error("Could not open " + std::string(lockdownmedldfilename));
	std::string lockdownmedemfilename;
	lockdownmedemfilename = "lockdownmedem.txt";
	auto& lockdownmedemfile = streams.output();
	lockdownmedemfile.open(lockdownmedemfilename);
	if (!lockdownmedemfile) throw std::runtime_error("Could not open " + std::string(lockdownmedemfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < lockdownmed.size(); i++) {
			lockdownmedldfile << lockdown[lockdownmed[i]][d] << "\t";
			lockdownmedemfile << excessmortalitypscoreinterpolated[lockdownmed[i]][d] << "\t";
		}
		lockdownmedldfile << std::endl;
		lockdownmedemfile << std::endl;
	}
	std::string lockdownhighldfilename;
	lockdownhighldfilename = "lockdownhighld.txt";
	auto& lockdownhighldfile = streams.output();
	lockdownhighldfile.open(lockdownhighldfilename);
	if (!lockdownhighldfile) throw std::runtime_error("Could not open " + std::string(lockdownhighldfilename));
	std::string lockdownhighemfilename;
	lockdownhighemfilename = "lockdownhighem.txt";
	auto& lockdownhighemfile = streams.output();
	lockdownhighemfile.open(lockdownhighemfilename);
	if (!lockdownhighemfile) throw std::runtime_error("Could not open " + std::string(lockdownhighemfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < lockdownhigh.size(); i++) {
			lockdownhighldfile << lockdown[lockdownhigh[i]][d] << "\t";
			lockdownhighemfile << excessmortalitypscoreinterpolated[lockdownhigh[i]][d] << "\t";
		}
		lockdownhighldfile << std::endl;
		lockdownhighemfile << std::endl;
	}
	std::cout << ".";



	// FACE MASKS
	std::vector<double> tempmasks;
	std::vector<int> maskslow;
	std::vector<int> masksmed;
	std::vector<int> maskshigh;
	double tm1, tm2;
	for (unsigned int i = 0; i < numentities; i++)
		if (cummasks[i] > 0 && !continent[i].empty()) // ignore all zero values and non countries
			tempmasks.push_back(cummasks[i]);
	if (tempmasks.size() < 3) throw std::runtime_error("Too few eligible countries for tertiles: tempmasks");
	std::sort(tempmasks.begin(), tempmasks.end());
	tm1 = tempmasks[std::round((tempmasks.size() / 3) - 1)];
	tm2 = tempmasks[std::round((2 * tempmasks.size() / 3) - 1)];
	for (unsigned int i = 0; i < numentities; i++)
		if (cummasks[i] > 0 && !continent[i].empty()) {
			if (cummasks[i] <= tm1)
				maskslow.push_back(i);
			else if (cummasks[i] <= tm2)
				masksmed.push_back(i);
			else
				maskshigh.push_back(i);
		}
	std::cout << ".";
	outputfile << "Countries with low cumulative face-covering policy index" << std::endl;
	outputfile << "Country\tCumulative face-covering policy (index-point days)" << std::endl;
	for (unsigned int i = 0; i < maskslow.size(); i++)
		display.push_back(entity[maskslow[i]] + "\t" + tostring(cummasks[maskslow[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();


	outputfile << "Countries with medium cumulative face-covering policy index" << std::endl;
	outputfile << "Country\tCumulative face-covering policy (index-point days)" << std::endl;
	for (unsigned int i = 0; i < masksmed.size(); i++)
		display.push_back(entity[masksmed[i]] + "\t" + tostring(cummasks[masksmed[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();

	outputfile << "Countries with high cumulative face-covering policy index" << std::endl;
	outputfile << "Country\tCumulative face-covering policy (index-point days)" << std::endl;
	for (unsigned int i = 0; i < maskshigh.size(); i++)
		display.push_back(entity[maskshigh[i]] + "\t" + tostring(cummasks[maskshigh[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();


	std::string maskslowmafilename;
	maskslowmafilename = "maskslowma.txt";
	auto& maskslowmafile = streams.output();
	maskslowmafile.open(maskslowmafilename);
	if (!maskslowmafile) throw std::runtime_error("Could not open " + std::string(maskslowmafilename));
	std::string maskslowemfilename;
	maskslowemfilename = "maskslowem.txt";
	auto& maskslowemfile = streams.output();
	maskslowemfile.open(maskslowemfilename);
	if (!maskslowemfile) throw std::runtime_error("Could not open " + std::string(maskslowemfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < maskslow.size(); i++) {
			maskslowmafile << masks[maskslow[i]][d] << "\t";
			maskslowemfile << excessmortalitypscoreinterpolated[maskslow[i]][d] << "\t";
		}
		maskslowmafile << std::endl;
		maskslowemfile << std::endl;
	}
	std::string masksmedmafilename;
	masksmedmafilename = "masksmedma.txt";
	auto& masksmedmafile = streams.output();
	masksmedmafile.open(masksmedmafilename);
	if (!masksmedmafile) throw std::runtime_error("Could not open " + std::string(masksmedmafilename));
	std::string masksmedemfilename;
	masksmedemfilename = "masksmedem.txt";
	auto& masksmedemfile = streams.output();
	masksmedemfile.open(masksmedemfilename);
	if (!masksmedemfile) throw std::runtime_error("Could not open " + std::string(masksmedemfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < masksmed.size(); i++) {
			masksmedmafile << masks[masksmed[i]][d] << "\t";
			masksmedemfile << excessmortalitypscoreinterpolated[masksmed[i]][d] << "\t";
		}
		masksmedmafile << std::endl;
		masksmedemfile << std::endl;
	}
	std::cout << ".";
	std::string maskshighmafilename;
	maskshighmafilename = "maskshighma.txt";
	auto& maskshighmafile = streams.output();
	maskshighmafile.open(maskshighmafilename);
	if (!maskshighmafile) throw std::runtime_error("Could not open " + std::string(maskshighmafilename));
	std::string maskshighemfilename;
	maskshighemfilename = "maskshighem.txt";
	auto& maskshighemfile = streams.output();
	maskshighemfile.open(maskshighemfilename);
	if (!maskshighemfile) throw std::runtime_error("Could not open " + std::string(maskshighemfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < maskshigh.size(); i++) {
			maskshighmafile << masks[maskshigh[i]][d] << "\t";
			maskshighemfile << excessmortalitypscoreinterpolated[maskshigh[i]][d] << "\t";
		}
		maskshighmafile << std::endl;
		maskshighemfile << std::endl;
	}
	std::cout << ".";







	// MASKS AND VACCINATIONS

	// maskslowvaxlow
	std::vector<int> maskslowvaxlow;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(maskslow.begin(), maskslow.end(), i) != maskslow.end()) && (std::find(vaxlow.begin(), vaxlow.end(), i) != vaxlow.end()))
			maskslowvaxlow.push_back(i);
	for (unsigned int i = 0; i < maskslowvaxlow.size(); i++)
		display.push_back(entity[maskslowvaxlow[i]] + "\t" + tostring(cummasks[maskslowvaxlow[i]]) + "\t" + tostring(cumvax[maskslowvaxlow[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with low cumulative face-covering policy index and low cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative face-covering policy (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string maskslowvaxlowfilename;
	maskslowvaxlowfilename = "maskslowvaxlow.txt";
	auto& maskslowvaxlowfile = streams.output();
	maskslowvaxlowfile.open(maskslowvaxlowfilename);
	if (!maskslowvaxlowfile) throw std::runtime_error("Could not open " + std::string(maskslowvaxlowfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < maskslowvaxlow.size(); i++)
			maskslowvaxlowfile << excessmortalitypscoreinterpolated[maskslowvaxlow[i]][d] << "\t";
		maskslowvaxlowfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// maskslowvaxmed
	std::vector<int> maskslowvaxmed;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(maskslow.begin(), maskslow.end(), i) != maskslow.end()) && (std::find(vaxmed.begin(), vaxmed.end(), i) != vaxmed.end()))
			maskslowvaxmed.push_back(i);
	for (unsigned int i = 0; i < maskslowvaxmed.size(); i++)
		display.push_back(entity[maskslowvaxmed[i]] + "\t" + tostring(cummasks[maskslowvaxmed[i]]) + "\t" + tostring(cumvax[maskslowvaxmed[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with low cumulative face-covering policy index and medium cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative face-covering policy (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string maskslowvaxmedfilename;
	maskslowvaxmedfilename = "maskslowvaxmed.txt";
	auto& maskslowvaxmedfile = streams.output();
	maskslowvaxmedfile.open(maskslowvaxmedfilename);
	if (!maskslowvaxmedfile) throw std::runtime_error("Could not open " + std::string(maskslowvaxmedfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < maskslowvaxmed.size(); i++)
			maskslowvaxmedfile << excessmortalitypscoreinterpolated[maskslowvaxmed[i]][d] << "\t";
		maskslowvaxmedfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// maskslowvaxhigh
	std::vector<int> maskslowvaxhigh;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(maskslow.begin(), maskslow.end(), i) != maskslow.end()) && (std::find(vaxhigh.begin(), vaxhigh.end(), i) != vaxhigh.end()))
			maskslowvaxhigh.push_back(i);
	for (unsigned int i = 0; i < maskslowvaxhigh.size(); i++)
		display.push_back(entity[maskslowvaxhigh[i]] + "\t" + tostring(cummasks[maskslowvaxhigh[i]]) + "\t" + tostring(cumvax[maskslowvaxhigh[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with low cumulative face-covering policy index and high cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative face-covering policy (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string maskslowvaxhighfilename;
	maskslowvaxhighfilename = "maskslowvaxhigh.txt";
	auto& maskslowvaxhighfile = streams.output();
	maskslowvaxhighfile.open(maskslowvaxhighfilename);
	if (!maskslowvaxhighfile) throw std::runtime_error("Could not open " + std::string(maskslowvaxhighfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < maskslowvaxhigh.size(); i++)
			maskslowvaxhighfile << excessmortalitypscoreinterpolated[maskslowvaxhigh[i]][d] << "\t";
		maskslowvaxhighfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// masksmedvaxlow
	std::vector<int> masksmedvaxlow;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(masksmed.begin(), masksmed.end(), i) != masksmed.end()) && (std::find(vaxlow.begin(), vaxlow.end(), i) != vaxlow.end()))
			masksmedvaxlow.push_back(i);
	for (unsigned int i = 0; i < masksmedvaxlow.size(); i++)
		display.push_back(entity[masksmedvaxlow[i]] + "\t" + tostring(cummasks[masksmedvaxlow[i]]) + "\t" + tostring(cumvax[masksmedvaxlow[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with medium cumulative face-covering policy index and low cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative face-covering policy (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string masksmedvaxlowfilename;
	masksmedvaxlowfilename = "masksmedvaxlow.txt";
	auto& masksmedvaxlowfile = streams.output();
	masksmedvaxlowfile.open(masksmedvaxlowfilename);
	if (!masksmedvaxlowfile) throw std::runtime_error("Could not open " + std::string(masksmedvaxlowfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < masksmedvaxlow.size(); i++)
			masksmedvaxlowfile << excessmortalitypscoreinterpolated[masksmedvaxlow[i]][d] << "\t";
		masksmedvaxlowfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// masksmedvaxmed
	std::vector<int> masksmedvaxmed;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(masksmed.begin(), masksmed.end(), i) != masksmed.end()) && (std::find(vaxmed.begin(), vaxmed.end(), i) != vaxmed.end()))
			masksmedvaxmed.push_back(i);
	for (unsigned int i = 0; i < masksmedvaxmed.size(); i++)
		display.push_back(entity[masksmedvaxmed[i]] + "\t" + tostring(cummasks[masksmedvaxmed[i]]) + "\t" + tostring(cumvax[masksmedvaxmed[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with medium cumulative face-covering policy index and medium cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative face-covering policy (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string masksmedvaxmedfilename;
	masksmedvaxmedfilename = "masksmedvaxmed.txt";
	auto& masksmedvaxmedfile = streams.output();
	masksmedvaxmedfile.open(masksmedvaxmedfilename);
	if (!masksmedvaxmedfile) throw std::runtime_error("Could not open " + std::string(masksmedvaxmedfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < masksmedvaxmed.size(); i++)
			masksmedvaxmedfile << excessmortalitypscoreinterpolated[masksmedvaxmed[i]][d] << "\t";
		masksmedvaxmedfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// masksmedvaxhigh
	std::vector<int> masksmedvaxhigh;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(masksmed.begin(), masksmed.end(), i) != masksmed.end()) && (std::find(vaxhigh.begin(), vaxhigh.end(), i) != vaxhigh.end()))
			masksmedvaxhigh.push_back(i);
	for (unsigned int i = 0; i < masksmedvaxhigh.size(); i++)
		display.push_back(entity[masksmedvaxhigh[i]] + "\t" + tostring(cummasks[masksmedvaxhigh[i]]) + "\t" + tostring(cumvax[masksmedvaxhigh[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with medium cumulative face-covering policy index and high cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative face-covering policy (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string masksmedvaxhighfilename;
	masksmedvaxhighfilename = "masksmedvaxhigh.txt";
	auto& masksmedvaxhighfile = streams.output();
	masksmedvaxhighfile.open(masksmedvaxhighfilename);
	if (!masksmedvaxhighfile) throw std::runtime_error("Could not open " + std::string(masksmedvaxhighfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < masksmedvaxhigh.size(); i++)
			masksmedvaxhighfile << excessmortalitypscoreinterpolated[masksmedvaxhigh[i]][d] << "\t";
		masksmedvaxhighfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// maskshighvaxlow
	std::vector<int> maskshighvaxlow;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(maskshigh.begin(), maskshigh.end(), i) != maskshigh.end()) && (std::find(vaxlow.begin(), vaxlow.end(), i) != vaxlow.end()))
			maskshighvaxlow.push_back(i);
	for (unsigned int i = 0; i < maskshighvaxlow.size(); i++)
		display.push_back(entity[maskshighvaxlow[i]] + "\t" + tostring(cummasks[maskshighvaxlow[i]]) + "\t" + tostring(cumvax[maskshighvaxlow[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with high cumulative face-covering policy index and low cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative face-covering policy (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string maskshighvaxlowfilename;
	maskshighvaxlowfilename = "maskshighvaxlow.txt";
	auto& maskshighvaxlowfile = streams.output();
	maskshighvaxlowfile.open(maskshighvaxlowfilename);
	if (!maskshighvaxlowfile) throw std::runtime_error("Could not open " + std::string(maskshighvaxlowfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < maskshighvaxlow.size(); i++)
			maskshighvaxlowfile << excessmortalitypscoreinterpolated[maskshighvaxlow[i]][d] << "\t";
		maskshighvaxlowfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// maskshighvaxmed
	std::vector<int> maskshighvaxmed;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(maskshigh.begin(), maskshigh.end(), i) != maskshigh.end()) && (std::find(vaxmed.begin(), vaxmed.end(), i) != vaxmed.end()))
			maskshighvaxmed.push_back(i);
	for (unsigned int i = 0; i < maskshighvaxmed.size(); i++)
		display.push_back(entity[maskshighvaxmed[i]] + "\t" + tostring(cummasks[maskshighvaxmed[i]]) + "\t" + tostring(cumvax[maskshighvaxmed[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with high cumulative face-covering policy index and medium cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative face-covering policy (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string maskshighvaxmedfilename;
	maskshighvaxmedfilename = "maskshighvaxmed.txt";
	auto& maskshighvaxmedfile = streams.output();
	maskshighvaxmedfile.open(maskshighvaxmedfilename);
	if (!maskshighvaxmedfile) throw std::runtime_error("Could not open " + std::string(maskshighvaxmedfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < maskshighvaxmed.size(); i++)
			maskshighvaxmedfile << excessmortalitypscoreinterpolated[maskshighvaxmed[i]][d] << "\t";
		maskshighvaxmedfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// maskshighvaxhigh
	std::vector<int> maskshighvaxhigh;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(maskshigh.begin(), maskshigh.end(), i) != maskshigh.end()) && (std::find(vaxhigh.begin(), vaxhigh.end(), i) != vaxhigh.end()))
			maskshighvaxhigh.push_back(i);
	for (unsigned int i = 0; i < maskshighvaxhigh.size(); i++)
		display.push_back(entity[maskshighvaxhigh[i]] + "\t" + tostring(cummasks[maskshighvaxhigh[i]]) + "\t" + tostring(cumvax[maskshighvaxhigh[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with high cumulative face-covering policy index and high cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative face-covering policy (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string maskshighvaxhighfilename;
	maskshighvaxhighfilename = "maskshighvaxhigh.txt";
	auto& maskshighvaxhighfile = streams.output();
	maskshighvaxhighfile.open(maskshighvaxhighfilename);
	if (!maskshighvaxhighfile) throw std::runtime_error("Could not open " + std::string(maskshighvaxhighfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < maskshighvaxhigh.size(); i++)
			maskshighvaxhighfile << excessmortalitypscoreinterpolated[maskshighvaxhigh[i]][d] << "\t";
		maskshighvaxhighfile << std::endl;
	}
	display.clear();
	std::cout << ".";












	// LOCKDOWN AND VACCINATIONS

	// lockdownlowvaxlow
	std::vector<int> lockdownlowvaxlow;
	for (unsigned int i = 0; i < numentities; i++)
	    if ((std::find(lockdownlow.begin(), lockdownlow.end(), i) != lockdownlow.end()) && (std::find(vaxlow.begin(), vaxlow.end(), i) != vaxlow.end()))
			lockdownlowvaxlow.push_back(i);
	for (unsigned int i = 0; i < lockdownlowvaxlow.size(); i++)
		display.push_back(entity[lockdownlowvaxlow[i]] + "\t" + tostring(cumlockdowns[lockdownlowvaxlow[i]]) + "\t" + tostring(cumvax[lockdownlowvaxlow[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with low cumulative stringency and low cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownlowvaxlowfilename;
	lockdownlowvaxlowfilename = "lockdownlowvaxlow.txt";
	auto& lockdownlowvaxlowfile = streams.output();
	lockdownlowvaxlowfile.open(lockdownlowvaxlowfilename);
	if (!lockdownlowvaxlowfile) throw std::runtime_error("Could not open " + std::string(lockdownlowvaxlowfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < lockdownlowvaxlow.size(); i++)
			lockdownlowvaxlowfile << excessmortalitypscoreinterpolated[lockdownlowvaxlow[i]][d] << "\t";
		lockdownlowvaxlowfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// lockdownlowvaxmed
	std::vector<int> lockdownlowvaxmed;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(lockdownlow.begin(), lockdownlow.end(), i) != lockdownlow.end()) && (std::find(vaxmed.begin(), vaxmed.end(), i) != vaxmed.end()))
			lockdownlowvaxmed.push_back(i);
	for (unsigned int i = 0; i < lockdownlowvaxmed.size(); i++)
		display.push_back(entity[lockdownlowvaxmed[i]] + "\t" + tostring(cumlockdowns[lockdownlowvaxmed[i]]) + "\t" + tostring(cumvax[lockdownlowvaxmed[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with low cumulative stringency and medium cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownlowvaxmedfilename;
	lockdownlowvaxmedfilename = "lockdownlowvaxmed.txt";
	auto& lockdownlowvaxmedfile = streams.output();
	lockdownlowvaxmedfile.open(lockdownlowvaxmedfilename);
	if (!lockdownlowvaxmedfile) throw std::runtime_error("Could not open " + std::string(lockdownlowvaxmedfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < lockdownlowvaxmed.size(); i++)
			lockdownlowvaxmedfile << excessmortalitypscoreinterpolated[lockdownlowvaxmed[i]][d] << "\t";
		lockdownlowvaxmedfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// lockdownlowvaxhigh
	std::vector<int> lockdownlowvaxhigh;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(lockdownlow.begin(), lockdownlow.end(), i) != lockdownlow.end()) && (std::find(vaxhigh.begin(), vaxhigh.end(), i) != vaxhigh.end()))
			lockdownlowvaxhigh.push_back(i);
	for (unsigned int i = 0; i < lockdownlowvaxhigh.size(); i++)
		display.push_back(entity[lockdownlowvaxhigh[i]] + "\t" + tostring(cumlockdowns[lockdownlowvaxhigh[i]]) + "\t" + tostring(cumvax[lockdownlowvaxhigh[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with low cumulative stringency and high cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownlowvaxhighfilename;
	lockdownlowvaxhighfilename = "lockdownlowvaxhigh.txt";
	auto& lockdownlowvaxhighfile = streams.output();
	lockdownlowvaxhighfile.open(lockdownlowvaxhighfilename);
	if (!lockdownlowvaxhighfile) throw std::runtime_error("Could not open " + std::string(lockdownlowvaxhighfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < lockdownlowvaxhigh.size(); i++)
			lockdownlowvaxhighfile << excessmortalitypscoreinterpolated[lockdownlowvaxhigh[i]][d] << "\t";
		lockdownlowvaxhighfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// lockdownmedvaxlow
	std::vector<int> lockdownmedvaxlow;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(lockdownmed.begin(), lockdownmed.end(), i) != lockdownmed.end()) && (std::find(vaxlow.begin(), vaxlow.end(), i) != vaxlow.end()))
			lockdownmedvaxlow.push_back(i);
	for (unsigned int i = 0; i < lockdownmedvaxlow.size(); i++)
		display.push_back(entity[lockdownmedvaxlow[i]] + "\t" + tostring(cumlockdowns[lockdownmedvaxlow[i]]) + "\t" + tostring(cumvax[lockdownmedvaxlow[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with medium cumulative stringency and low cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownmedvaxlowfilename;
	lockdownmedvaxlowfilename = "lockdownmedvaxlow.txt";
	auto& lockdownmedvaxlowfile = streams.output();
	lockdownmedvaxlowfile.open(lockdownmedvaxlowfilename);
	if (!lockdownmedvaxlowfile) throw std::runtime_error("Could not open " + std::string(lockdownmedvaxlowfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < lockdownmedvaxlow.size(); i++)
			lockdownmedvaxlowfile << excessmortalitypscoreinterpolated[lockdownmedvaxlow[i]][d] << "\t";
		lockdownmedvaxlowfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// lockdownmedvaxmed
	std::vector<int> lockdownmedvaxmed;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(lockdownmed.begin(), lockdownmed.end(), i) != lockdownmed.end()) && (std::find(vaxmed.begin(), vaxmed.end(), i) != vaxmed.end()))
			lockdownmedvaxmed.push_back(i);
	for (unsigned int i = 0; i < lockdownmedvaxmed.size(); i++)
		display.push_back(entity[lockdownmedvaxmed[i]] + "\t" + tostring(cumlockdowns[lockdownmedvaxmed[i]]) + "\t" + tostring(cumvax[lockdownmedvaxmed[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with medium cumulative stringency and medium cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownmedvaxmedfilename;
	lockdownmedvaxmedfilename = "lockdownmedvaxmed.txt";
	auto& lockdownmedvaxmedfile = streams.output();
	lockdownmedvaxmedfile.open(lockdownmedvaxmedfilename);
	if (!lockdownmedvaxmedfile) throw std::runtime_error("Could not open " + std::string(lockdownmedvaxmedfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < lockdownmedvaxmed.size(); i++)
			lockdownmedvaxmedfile << excessmortalitypscoreinterpolated[lockdownmedvaxmed[i]][d] << "\t";
		lockdownmedvaxmedfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// lockdownmedvaxhigh
	std::vector<int> lockdownmedvaxhigh;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(lockdownmed.begin(), lockdownmed.end(), i) != lockdownmed.end()) && (std::find(vaxhigh.begin(), vaxhigh.end(), i) != vaxhigh.end()))
			lockdownmedvaxhigh.push_back(i);
	for (unsigned int i = 0; i < lockdownmedvaxhigh.size(); i++)
		display.push_back(entity[lockdownmedvaxhigh[i]] + "\t" + tostring(cumlockdowns[lockdownmedvaxhigh[i]]) + "\t" + tostring(cumvax[lockdownmedvaxhigh[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with medium cumulative stringency and high cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownmedvaxhighfilename;
	lockdownmedvaxhighfilename = "lockdownmedvaxhigh.txt";
	auto& lockdownmedvaxhighfile = streams.output();
	lockdownmedvaxhighfile.open(lockdownmedvaxhighfilename);
	if (!lockdownmedvaxhighfile) throw std::runtime_error("Could not open " + std::string(lockdownmedvaxhighfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < lockdownmedvaxhigh.size(); i++)
			lockdownmedvaxhighfile << excessmortalitypscoreinterpolated[lockdownmedvaxhigh[i]][d] << "\t";
		lockdownmedvaxhighfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// lockdownhighvaxlow
	std::vector<int> lockdownhighvaxlow;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(lockdownhigh.begin(), lockdownhigh.end(), i) != lockdownhigh.end()) && (std::find(vaxlow.begin(), vaxlow.end(), i) != vaxlow.end()))
			lockdownhighvaxlow.push_back(i);
	for (unsigned int i = 0; i < lockdownhighvaxlow.size(); i++)
		display.push_back(entity[lockdownhighvaxlow[i]] + "\t" + tostring(cumlockdowns[lockdownhighvaxlow[i]]) + "\t" + tostring(cumvax[lockdownhighvaxlow[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with high cumulative stringency and low cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownhighvaxlowfilename;
	lockdownhighvaxlowfilename = "lockdownhighvaxlow.txt";
	auto& lockdownhighvaxlowfile = streams.output();
	lockdownhighvaxlowfile.open(lockdownhighvaxlowfilename);
	if (!lockdownhighvaxlowfile) throw std::runtime_error("Could not open " + std::string(lockdownhighvaxlowfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < lockdownhighvaxlow.size(); i++)
			lockdownhighvaxlowfile << excessmortalitypscoreinterpolated[lockdownhighvaxlow[i]][d] << "\t";
		lockdownhighvaxlowfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// lockdownhighvaxmed
	std::vector<int> lockdownhighvaxmed;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(lockdownhigh.begin(), lockdownhigh.end(), i) != lockdownhigh.end()) && (std::find(vaxmed.begin(), vaxmed.end(), i) != vaxmed.end()))
			lockdownhighvaxmed.push_back(i);
	for (unsigned int i = 0; i < lockdownhighvaxmed.size(); i++)
		display.push_back(entity[lockdownhighvaxmed[i]] + "\t" + tostring(cumlockdowns[lockdownhighvaxmed[i]]) + "\t" + tostring(cumvax[lockdownhighvaxmed[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with high cumulative stringency and medium cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownhighvaxmedfilename;
	lockdownhighvaxmedfilename = "lockdownhighvaxmed.txt";
	auto& lockdownhighvaxmedfile = streams.output();
	lockdownhighvaxmedfile.open(lockdownhighvaxmedfilename);
	if (!lockdownhighvaxmedfile) throw std::runtime_error("Could not open " + std::string(lockdownhighvaxmedfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < lockdownhighvaxmed.size(); i++)
			lockdownhighvaxmedfile << excessmortalitypscoreinterpolated[lockdownhighvaxmed[i]][d] << "\t";
		lockdownhighvaxmedfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// lockdownhighvaxhigh
	std::vector<int> lockdownhighvaxhigh;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(lockdownhigh.begin(), lockdownhigh.end(), i) != lockdownhigh.end()) && (std::find(vaxhigh.begin(), vaxhigh.end(), i) != vaxhigh.end()))
			lockdownhighvaxhigh.push_back(i);
	for (unsigned int i = 0; i < lockdownhighvaxhigh.size(); i++)
		display.push_back(entity[lockdownhighvaxhigh[i]] + "\t" + tostring(cumlockdowns[lockdownhighvaxhigh[i]]) + "\t" + tostring(cumvax[lockdownhighvaxhigh[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with high cumulative stringency and high cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownhighvaxhighfilename;
	lockdownhighvaxhighfilename = "lockdownhighvaxhigh.txt";
	auto& lockdownhighvaxhighfile = streams.output();
	lockdownhighvaxhighfile.open(lockdownhighvaxhighfilename);
	if (!lockdownhighvaxhighfile) throw std::runtime_error("Could not open " + std::string(lockdownhighvaxhighfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < lockdownhighvaxhigh.size(); i++)
			lockdownhighvaxhighfile << excessmortalitypscoreinterpolated[lockdownhighvaxhigh[i]][d] << "\t";
		lockdownhighvaxhighfile << std::endl;
	}
	display.clear();
	std::cout << ".";



	// COVIDDEATHS AND VACCINATIONS

	// coviddeathslowvaxlow
	std::vector<int> coviddeathslowvaxlow;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(coviddeathslow.begin(), coviddeathslow.end(), i) != coviddeathslow.end()) && (std::find(vaxlow.begin(), vaxlow.end(), i) != vaxlow.end()))
			coviddeathslowvaxlow.push_back(i);
	for (unsigned int i = 0; i < coviddeathslowvaxlow.size(); i++)
		display.push_back(entity[coviddeathslowvaxlow[i]] + "\t" + tostring(cumcoviddeaths[coviddeathslowvaxlow[i]]) + "\t" + tostring(cumvax[coviddeathslowvaxlow[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with low cumulative COVID-19 deaths per million and low cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathslowvaxlowfilename;
	coviddeathslowvaxlowfilename = "coviddeathslowvaxlow.txt";
	auto& coviddeathslowvaxlowfile = streams.output();
	coviddeathslowvaxlowfile.open(coviddeathslowvaxlowfilename);
	if (!coviddeathslowvaxlowfile) throw std::runtime_error("Could not open " + std::string(coviddeathslowvaxlowfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathslowvaxlow.size(); i++)
			coviddeathslowvaxlowfile << excessmortalitypscoreinterpolated[coviddeathslowvaxlow[i]][d] << "\t";
		coviddeathslowvaxlowfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// coviddeathslowvaxmed
	std::vector<int> coviddeathslowvaxmed;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(coviddeathslow.begin(), coviddeathslow.end(), i) != coviddeathslow.end()) && (std::find(vaxmed.begin(), vaxmed.end(), i) != vaxmed.end()))
			coviddeathslowvaxmed.push_back(i);
	for (unsigned int i = 0; i < coviddeathslowvaxmed.size(); i++)
		display.push_back(entity[coviddeathslowvaxmed[i]] + "\t" + tostring(cumcoviddeaths[coviddeathslowvaxmed[i]]) + "\t" + tostring(cumvax[coviddeathslowvaxmed[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with low cumulative COVID-19 deaths per million and medium cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathslowvaxmedfilename;
	coviddeathslowvaxmedfilename = "coviddeathslowvaxmed.txt";
	auto& coviddeathslowvaxmedfile = streams.output();
	coviddeathslowvaxmedfile.open(coviddeathslowvaxmedfilename);
	if (!coviddeathslowvaxmedfile) throw std::runtime_error("Could not open " + std::string(coviddeathslowvaxmedfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathslowvaxmed.size(); i++)
			coviddeathslowvaxmedfile << excessmortalitypscoreinterpolated[coviddeathslowvaxmed[i]][d] << "\t";
		coviddeathslowvaxmedfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// coviddeathslowvaxhigh
	std::vector<int> coviddeathslowvaxhigh;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(coviddeathslow.begin(), coviddeathslow.end(), i) != coviddeathslow.end()) && (std::find(vaxhigh.begin(), vaxhigh.end(), i) != vaxhigh.end()))
			coviddeathslowvaxhigh.push_back(i);
	for (unsigned int i = 0; i < coviddeathslowvaxhigh.size(); i++)
		display.push_back(entity[coviddeathslowvaxhigh[i]] + "\t" + tostring(cumcoviddeaths[coviddeathslowvaxhigh[i]]) + "\t" + tostring(cumvax[coviddeathslowvaxhigh[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with low cumulative COVID-19 deaths per million and high cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathslowvaxhighfilename;
	coviddeathslowvaxhighfilename = "coviddeathslowvaxhigh.txt";
	auto& coviddeathslowvaxhighfile = streams.output();
	coviddeathslowvaxhighfile.open(coviddeathslowvaxhighfilename);
	if (!coviddeathslowvaxhighfile) throw std::runtime_error("Could not open " + std::string(coviddeathslowvaxhighfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathslowvaxhigh.size(); i++)
			coviddeathslowvaxhighfile << excessmortalitypscoreinterpolated[coviddeathslowvaxhigh[i]][d] << "\t";
		coviddeathslowvaxhighfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// coviddeathsmedvaxlow
	std::vector<int> coviddeathsmedvaxlow;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(coviddeathsmed.begin(), coviddeathsmed.end(), i) != coviddeathsmed.end()) && (std::find(vaxlow.begin(), vaxlow.end(), i) != vaxlow.end()))
			coviddeathsmedvaxlow.push_back(i);
	for (unsigned int i = 0; i < coviddeathsmedvaxlow.size(); i++)
		display.push_back(entity[coviddeathsmedvaxlow[i]] + "\t" + tostring(cumcoviddeaths[coviddeathsmedvaxlow[i]]) + "\t" + tostring(cumvax[coviddeathsmedvaxlow[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with medium cumulative COVID-19 deaths per million and low cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathsmedvaxlowfilename;
	coviddeathsmedvaxlowfilename = "coviddeathsmedvaxlow.txt";
	auto& coviddeathsmedvaxlowfile = streams.output();
	coviddeathsmedvaxlowfile.open(coviddeathsmedvaxlowfilename);
	if (!coviddeathsmedvaxlowfile) throw std::runtime_error("Could not open " + std::string(coviddeathsmedvaxlowfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathsmedvaxlow.size(); i++)
			coviddeathsmedvaxlowfile << excessmortalitypscoreinterpolated[coviddeathsmedvaxlow[i]][d] << "\t";
		coviddeathsmedvaxlowfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// coviddeathsmedvaxmed
	std::vector<int> coviddeathsmedvaxmed;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(coviddeathsmed.begin(), coviddeathsmed.end(), i) != coviddeathsmed.end()) && (std::find(vaxmed.begin(), vaxmed.end(), i) != vaxmed.end()))
			coviddeathsmedvaxmed.push_back(i);
	for (unsigned int i = 0; i < coviddeathsmedvaxmed.size(); i++)
		display.push_back(entity[coviddeathsmedvaxmed[i]] + "\t" + tostring(cumcoviddeaths[coviddeathsmedvaxmed[i]]) + "\t" + tostring(cumvax[coviddeathsmedvaxmed[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with medium cumulative COVID-19 deaths per million and medium cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathsmedvaxmedfilename;
	coviddeathsmedvaxmedfilename = "coviddeathsmedvaxmed.txt";
	auto& coviddeathsmedvaxmedfile = streams.output();
	coviddeathsmedvaxmedfile.open(coviddeathsmedvaxmedfilename);
	if (!coviddeathsmedvaxmedfile) throw std::runtime_error("Could not open " + std::string(coviddeathsmedvaxmedfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathsmedvaxmed.size(); i++)
			coviddeathsmedvaxmedfile << excessmortalitypscoreinterpolated[coviddeathsmedvaxmed[i]][d] << "\t";
		coviddeathsmedvaxmedfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// coviddeathsmedvaxhigh
	std::vector<int> coviddeathsmedvaxhigh;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(coviddeathsmed.begin(), coviddeathsmed.end(), i) != coviddeathsmed.end()) && (std::find(vaxhigh.begin(), vaxhigh.end(), i) != vaxhigh.end()))
			coviddeathsmedvaxhigh.push_back(i);
	for (unsigned int i = 0; i < coviddeathsmedvaxhigh.size(); i++)
		display.push_back(entity[coviddeathsmedvaxhigh[i]] + "\t" + tostring(cumcoviddeaths[coviddeathsmedvaxhigh[i]]) + "\t" + tostring(cumvax[coviddeathsmedvaxhigh[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with medium cumulative COVID-19 deaths per million and high cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathsmedvaxhighfilename;
	coviddeathsmedvaxhighfilename = "coviddeathsmedvaxhigh.txt";
	auto& coviddeathsmedvaxhighfile = streams.output();
	coviddeathsmedvaxhighfile.open(coviddeathsmedvaxhighfilename);
	if (!coviddeathsmedvaxhighfile) throw std::runtime_error("Could not open " + std::string(coviddeathsmedvaxhighfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathsmedvaxhigh.size(); i++)
			coviddeathsmedvaxhighfile << excessmortalitypscoreinterpolated[coviddeathsmedvaxhigh[i]][d] << "\t";
		coviddeathsmedvaxhighfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// coviddeathshighvaxlow
	std::vector<int> coviddeathshighvaxlow;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(coviddeathshigh.begin(), coviddeathshigh.end(), i) != coviddeathshigh.end()) && (std::find(vaxlow.begin(), vaxlow.end(), i) != vaxlow.end()))
			coviddeathshighvaxlow.push_back(i);
	for (unsigned int i = 0; i < coviddeathshighvaxlow.size(); i++)
		display.push_back(entity[coviddeathshighvaxlow[i]] + "\t" + tostring(cumcoviddeaths[coviddeathshighvaxlow[i]]) + "\t" + tostring(cumvax[coviddeathshighvaxlow[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with high cumulative COVID-19 deaths per million and low cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathshighvaxlowfilename;
	coviddeathshighvaxlowfilename = "coviddeathshighvaxlow.txt";
	auto& coviddeathshighvaxlowfile = streams.output();
	coviddeathshighvaxlowfile.open(coviddeathshighvaxlowfilename);
	if (!coviddeathshighvaxlowfile) throw std::runtime_error("Could not open " + std::string(coviddeathshighvaxlowfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathshighvaxlow.size(); i++)
			coviddeathshighvaxlowfile << excessmortalitypscoreinterpolated[coviddeathshighvaxlow[i]][d] << "\t";
		coviddeathshighvaxlowfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// coviddeathshighvaxmed
	std::vector<int> coviddeathshighvaxmed;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(coviddeathshigh.begin(), coviddeathshigh.end(), i) != coviddeathshigh.end()) && (std::find(vaxmed.begin(), vaxmed.end(), i) != vaxmed.end()))
			coviddeathshighvaxmed.push_back(i);
	for (unsigned int i = 0; i < coviddeathshighvaxmed.size(); i++)
		display.push_back(entity[coviddeathshighvaxmed[i]] + "\t" + tostring(cumcoviddeaths[coviddeathshighvaxmed[i]]) + "\t" + tostring(cumvax[coviddeathshighvaxmed[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with high cumulative COVID-19 deaths per million and medium cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathshighvaxmedfilename;
	coviddeathshighvaxmedfilename = "coviddeathshighvaxmed.txt";
	auto& coviddeathshighvaxmedfile = streams.output();
	coviddeathshighvaxmedfile.open(coviddeathshighvaxmedfilename);
	if (!coviddeathshighvaxmedfile) throw std::runtime_error("Could not open " + std::string(coviddeathshighvaxmedfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathshighvaxmed.size(); i++)
			coviddeathshighvaxmedfile << excessmortalitypscoreinterpolated[coviddeathshighvaxmed[i]][d] << "\t";
		coviddeathshighvaxmedfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// coviddeathshighvaxhigh
	std::vector<int> coviddeathshighvaxhigh;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(coviddeathshigh.begin(), coviddeathshigh.end(), i) != coviddeathshigh.end()) && (std::find(vaxhigh.begin(), vaxhigh.end(), i) != vaxhigh.end()))
			coviddeathshighvaxhigh.push_back(i);
	for (unsigned int i = 0; i < coviddeathshighvaxhigh.size(); i++)
		display.push_back(entity[coviddeathshighvaxhigh[i]] + "\t" + tostring(cumcoviddeaths[coviddeathshighvaxhigh[i]]) + "\t" + tostring(cumvax[coviddeathshighvaxhigh[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with high cumulative COVID-19 deaths per million and high cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathshighvaxhighfilename;
	coviddeathshighvaxhighfilename = "coviddeathshighvaxhigh.txt";
	auto& coviddeathshighvaxhighfile = streams.output();
	coviddeathshighvaxhighfile.open(coviddeathshighvaxhighfilename);
	if (!coviddeathshighvaxhighfile) throw std::runtime_error("Could not open " + std::string(coviddeathshighvaxhighfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathshighvaxhigh.size(); i++)
			coviddeathshighvaxhighfile << excessmortalitypscoreinterpolated[coviddeathshighvaxhigh[i]][d] << "\t";
		coviddeathshighvaxhighfile << std::endl;
	}
	display.clear();
	std::cout << ".";






	// THREE FACTORS

	// coviddeathslowlockdownlowvaxlow
	std::vector<int> coviddeathslowlockdownlowvaxlow;
	for (unsigned int i = 0; i < numentities; i++)
	    if ((std::find(coviddeathslow.begin(), coviddeathslow.end(), i) != coviddeathslow.end()) && (std::find(lockdownlow.begin(), lockdownlow.end(), i) != lockdownlow.end()) && (std::find(vaxlow.begin(), vaxlow.end(), i) != vaxlow.end()))
		    coviddeathslowlockdownlowvaxlow.push_back(i);
	for (unsigned int i = 0; i < coviddeathslowlockdownlowvaxlow.size(); i++)
		display.push_back(entity[coviddeathslowlockdownlowvaxlow[i]] + "\t" + tostring(cumcoviddeaths[coviddeathslowlockdownlowvaxlow[i]]) + "\t" + tostring(cumlockdowns[coviddeathslowlockdownlowvaxlow[i]]) + "\t" + tostring(cumvax[coviddeathslowlockdownlowvaxlow[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with low cumulative COVID-19 deaths per million, low cumulative stringency and low cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tCumulative lockdown stringency (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathslowlockdownlowvaxlowfilename;
	coviddeathslowlockdownlowvaxlowfilename = "coviddeathslowlockdownlowvaxlow.txt";
	auto& coviddeathslowlockdownlowvaxlowfile = streams.output();
	coviddeathslowlockdownlowvaxlowfile.open(coviddeathslowlockdownlowvaxlowfilename);
	if (!coviddeathslowlockdownlowvaxlowfile) throw std::runtime_error("Could not open " + std::string(coviddeathslowlockdownlowvaxlowfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathslowlockdownlowvaxlow.size(); i++)
			coviddeathslowlockdownlowvaxlowfile << excessmortalitypscoreinterpolated[coviddeathslowlockdownlowvaxlow[i]][d] << "\t";
		coviddeathslowlockdownlowvaxlowfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// coviddeathshighlockdownlowvaxlow
	std::vector<int> coviddeathshighlockdownlowvaxlow;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(coviddeathshigh.begin(), coviddeathshigh.end(), i) != coviddeathshigh.end()) && (std::find(lockdownlow.begin(), lockdownlow.end(), i) != lockdownlow.end()) && (std::find(vaxlow.begin(), vaxlow.end(), i) != vaxlow.end()))
		    coviddeathshighlockdownlowvaxlow.push_back(i);
	for (unsigned int i = 0; i < coviddeathshighlockdownlowvaxlow.size(); i++)
		display.push_back(entity[coviddeathshighlockdownlowvaxlow[i]] + "\t" + tostring(cumcoviddeaths[coviddeathshighlockdownlowvaxlow[i]]) + "\t" + tostring(cumlockdowns[coviddeathshighlockdownlowvaxlow[i]]) + "\t" + tostring(cumvax[coviddeathshighlockdownlowvaxlow[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with high cumulative COVID-19 deaths per million, low cumulative stringency and low cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tCumulative lockdown stringency (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathshighlockdownlowvaxlowfilename;
	coviddeathshighlockdownlowvaxlowfilename = "coviddeathshighlockdownlowvaxlow.txt";
	auto& coviddeathshighlockdownlowvaxlowfile = streams.output();
	coviddeathshighlockdownlowvaxlowfile.open(coviddeathshighlockdownlowvaxlowfilename);
	if (!coviddeathshighlockdownlowvaxlowfile) throw std::runtime_error("Could not open " + std::string(coviddeathshighlockdownlowvaxlowfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathshighlockdownlowvaxlow.size(); i++)
			coviddeathshighlockdownlowvaxlowfile << excessmortalitypscoreinterpolated[coviddeathshighlockdownlowvaxlow[i]][d] << "\t";
		coviddeathshighlockdownlowvaxlowfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// coviddeathslowlockdownhighvaxlow
	std::vector<int> coviddeathslowlockdownhighvaxlow;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(coviddeathslow.begin(), coviddeathslow.end(), i) != coviddeathslow.end()) && (std::find(lockdownhigh.begin(), lockdownhigh.end(), i) != lockdownhigh.end()) && (std::find(vaxlow.begin(), vaxlow.end(), i) != vaxlow.end()))
			coviddeathslowlockdownhighvaxlow.push_back(i);
	for (unsigned int i = 0; i < coviddeathslowlockdownhighvaxlow.size(); i++)
		display.push_back(entity[coviddeathslowlockdownhighvaxlow[i]] + "\t" + tostring(cumcoviddeaths[coviddeathslowlockdownhighvaxlow[i]]) + "\t" + tostring(cumlockdowns[coviddeathslowlockdownhighvaxlow[i]]) + "\t" + tostring(cumvax[coviddeathslowlockdownhighvaxlow[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with low cumulative COVID-19 deaths per million, high cumulative stringency and low cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tCumulative lockdown stringency (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathslowlockdownhighvaxlowfilename;
	coviddeathslowlockdownhighvaxlowfilename = "coviddeathslowlockdownhighvaxlow.txt";
	auto& coviddeathslowlockdownhighvaxlowfile = streams.output();
	coviddeathslowlockdownhighvaxlowfile.open(coviddeathslowlockdownhighvaxlowfilename);
	if (!coviddeathslowlockdownhighvaxlowfile) throw std::runtime_error("Could not open " + std::string(coviddeathslowlockdownhighvaxlowfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathslowlockdownhighvaxlow.size(); i++)
			coviddeathslowlockdownhighvaxlowfile << excessmortalitypscoreinterpolated[coviddeathslowlockdownhighvaxlow[i]][d] << "\t";
		coviddeathslowlockdownhighvaxlowfile << std::endl;
	}
	display.clear();
	std::cout << ".";

	// coviddeathslowlockdownlowvaxhigh
	std::vector<int> coviddeathslowlockdownlowvaxhigh;
	for (unsigned int i = 0; i < numentities; i++)
		if ((std::find(coviddeathslow.begin(), coviddeathslow.end(), i) != coviddeathslow.end()) && (std::find(lockdownlow.begin(), lockdownlow.end(), i) != lockdownlow.end()) && (std::find(vaxhigh.begin(), vaxhigh.end(), i) != vaxhigh.end()))
			coviddeathslowlockdownlowvaxhigh.push_back(i);
	for (unsigned int i = 0; i < coviddeathslowlockdownlowvaxhigh.size(); i++)
		display.push_back(entity[coviddeathslowlockdownlowvaxhigh[i]] + "\t" + tostring(cumcoviddeaths[coviddeathslowlockdownlowvaxhigh[i]]) + "\t" + tostring(cumlockdowns[coviddeathslowlockdownlowvaxhigh[i]]) + "\t" + tostring(cumvax[coviddeathslowlockdownlowvaxhigh[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with low cumulative COVID-19 deaths per million, low cumulative stringency and high cumulative vaccine doses per 100 people" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tCumulative lockdown stringency (index-point days)\tTotal vaccine doses per 100 people" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathslowlockdownlowvaxhighfilename;
	coviddeathslowlockdownlowvaxhighfilename = "coviddeathslowlockdownlowvaxhigh.txt";
	auto& coviddeathslowlockdownlowvaxhighfile = streams.output();
	coviddeathslowlockdownlowvaxhighfile.open(coviddeathslowlockdownlowvaxhighfilename);
	if (!coviddeathslowlockdownlowvaxhighfile) throw std::runtime_error("Could not open " + std::string(coviddeathslowlockdownlowvaxhighfilename));
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < coviddeathslowlockdownlowvaxhigh.size(); i++)
			coviddeathslowlockdownlowvaxhighfile << excessmortalitypscoreinterpolated[coviddeathslowlockdownlowvaxhigh[i]][d] << "\t";
		coviddeathslowlockdownlowvaxhighfile << std::endl;
	}
	display.clear();
	std::cout << ".";























	// INDIVIDUAL COUNTRIES

	std::string country;
	std::string countrylistfilename;
	countrylistfilename = "countrylist.txt";
	auto& countrylistfile = streams.output();
	countrylistfile.open(countrylistfilename);
	if (!countrylistfile) throw std::runtime_error("Could not open " + std::string(countrylistfilename));

	std::string countryfilename;
	auto& countryfile = streams.output();
	for (int c = 0; c < entity.size(); c++) {
		country = entity[c];
		countrylistfile << country << std::endl;
		std::replace(country.begin(), country.end(), ' ', '_');
		countryfilename = country + ".txt";
		countryfile.open(countryfilename);
		if (!countryfile) throw std::runtime_error("Could not open " + std::string(countryfilename));
		for (unsigned int d = 0; d < numdates; d++) {
			date_duration dd(d);
			date d2 = firstdate + dd;
			if (d2.day() < 10)
				dp = "0";
			else
				dp = "";
			if (d2.month() < 10)
				mp = "0";
			else
				mp = "";
			countryfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << lockdown[c][d] << "\t" << masks[c][d] << "\t" << vaccinations[c][d] << "\t" << coviddeaths[c][d] << "\t" << excessmortalitypscoreinterpolated[c][d] << std::endl;
		}
		countryfile.clear();
		countryfile.close();
	}
	countrylistfile.close();
	std::cout << ".";





    // COUNTRY PAIRS: each helper owns and closes one output stream.
    write_country_pair("AlgeriaEgyptCD.txt", firstdate, vaccinations[CountryIndex(code, "DZA")], vaccinations[CountryIndex(code, "EGY")], coviddeaths[CountryIndex(code, "DZA")], coviddeaths[CountryIndex(code, "EGY")]);
    write_country_pair("AlgeriaEgyptEM.txt", firstdate, vaccinations[CountryIndex(code, "DZA")], vaccinations[CountryIndex(code, "EGY")], excessmortalitypscoreinterpolated[CountryIndex(code, "DZA")], excessmortalitypscoreinterpolated[CountryIndex(code, "EGY")]);
    write_country_pair("AntiguaandBarbudaCubaEM.txt", firstdate, vaccinations[CountryIndex(code, "ATG")], vaccinations[CountryIndex(code, "CUB")], excessmortalitypscoreinterpolated[CountryIndex(code, "ATG")], excessmortalitypscoreinterpolated[CountryIndex(code, "CUB")]);
    write_country_pair("ArgentinaParaguayEM.txt", firstdate, vaccinations[CountryIndex(code, "ARG")], vaccinations[CountryIndex(code, "PRY")], excessmortalitypscoreinterpolated[CountryIndex(code, "ARG")], excessmortalitypscoreinterpolated[CountryIndex(code, "PRY")]);
    write_country_pair("ArmeniaAzerbaijanCD.txt", firstdate, vaccinations[CountryIndex(code, "ARM")], vaccinations[CountryIndex(code, "AZE")], coviddeaths[CountryIndex(code, "ARM")], coviddeaths[CountryIndex(code, "AZE")]);
    write_country_pair("ArmeniaAzerbaijanEM.txt", firstdate, vaccinations[CountryIndex(code, "ARM")], vaccinations[CountryIndex(code, "AZE")], excessmortalitypscoreinterpolated[CountryIndex(code, "ARM")], excessmortalitypscoreinterpolated[CountryIndex(code, "AZE")]);
    write_country_pair("AustraliaFrenchPolynesiaEM.txt", firstdate, vaccinations[CountryIndex(code, "AUS")], vaccinations[CountryIndex(code, "PYF")], excessmortalitypscoreinterpolated[CountryIndex(code, "AUS")], excessmortalitypscoreinterpolated[CountryIndex(code, "PYF")]);
    write_country_pair("AustraliaNewCaledoniaEM.txt", firstdate, vaccinations[CountryIndex(code, "AUS")], vaccinations[CountryIndex(code, "NCL")], excessmortalitypscoreinterpolated[CountryIndex(code, "AUS")], excessmortalitypscoreinterpolated[CountryIndex(code, "NCL")]);
    write_country_pair("AustraliaNewZealandCD.txt", firstdate, vaccinations[CountryIndex(code, "AUS")], vaccinations[CountryIndex(code, "NZL")], coviddeaths[CountryIndex(code, "AUS")], coviddeaths[CountryIndex(code, "NZL")]);
    write_country_pair("AustraliaNewZealandEM.txt", firstdate, vaccinations[CountryIndex(code, "AUS")], vaccinations[CountryIndex(code, "NZL")], excessmortalitypscoreinterpolated[CountryIndex(code, "AUS")], excessmortalitypscoreinterpolated[CountryIndex(code, "NZL")]);
    write_country_pair("AustraliaMalaysiaEM.txt", firstdate, vaccinations[CountryIndex(code, "AUS")], vaccinations[CountryIndex(code, "MYS")], excessmortalitypscoreinterpolated[CountryIndex(code, "AUS")], excessmortalitypscoreinterpolated[CountryIndex(code, "MYS")]);
    write_country_pair("BelarusDenmarkCD.txt", firstdate, vaccinations[CountryIndex(code, "BLR")], vaccinations[CountryIndex(code, "DNK")], coviddeaths[CountryIndex(code, "BLR")], coviddeaths[CountryIndex(code, "DNK")]);
    write_country_pair("BelarusDenmarkEM.txt", firstdate, vaccinations[CountryIndex(code, "BLR")], vaccinations[CountryIndex(code, "DNK")], excessmortalitypscoreinterpolated[CountryIndex(code, "BLR")], excessmortalitypscoreinterpolated[CountryIndex(code, "DNK")]);
    write_country_pair("BhutanSingaporeCD.txt", firstdate, vaccinations[CountryIndex(code, "BTN")], vaccinations[CountryIndex(code, "SGP")], coviddeaths[CountryIndex(code, "BTN")], coviddeaths[CountryIndex(code, "SGP")]);
    write_country_pair("BhutanSingaporeEM.txt", firstdate, vaccinations[CountryIndex(code, "BTN")], vaccinations[CountryIndex(code, "SGP")], excessmortalitypscoreinterpolated[CountryIndex(code, "BTN")], excessmortalitypscoreinterpolated[CountryIndex(code, "SGP")]);
    write_country_pair("BhutanThailandCD.txt", firstdate, vaccinations[CountryIndex(code, "BTN")], vaccinations[CountryIndex(code, "THA")], coviddeaths[CountryIndex(code, "BTN")], coviddeaths[CountryIndex(code, "THA")]);
    write_country_pair("BhutanThailandEM.txt", firstdate, vaccinations[CountryIndex(code, "BTN")], vaccinations[CountryIndex(code, "THA")], excessmortalitypscoreinterpolated[CountryIndex(code, "BTN")], excessmortalitypscoreinterpolated[CountryIndex(code, "THA")]);
    write_country_pair("BhutanVietnamCD.txt", firstdate, vaccinations[CountryIndex(code, "BTN")], vaccinations[CountryIndex(code, "VNM")], coviddeaths[CountryIndex(code, "BTN")], coviddeaths[CountryIndex(code, "VNM")]);
    write_country_pair("BurundiEritreaCD.txt", firstdate, vaccinations[CountryIndex(code, "BDI")], vaccinations[CountryIndex(code, "ERI")], coviddeaths[CountryIndex(code, "BDI")], coviddeaths[CountryIndex(code, "ERI")]);
    write_country_pair("BosniaandHerzegovinaRomaniaEM.txt", firstdate, vaccinations[CountryIndex(code, "BIH")], vaccinations[CountryIndex(code, "ROU")], excessmortalitypscoreinterpolated[CountryIndex(code, "BIH")], excessmortalitypscoreinterpolated[CountryIndex(code, "ROU")]);
    write_country_pair("BulgariaSerbiaEM.txt", firstdate, vaccinations[CountryIndex(code, "BGR")], vaccinations[CountryIndex(code, "SRB")], excessmortalitypscoreinterpolated[CountryIndex(code, "BGR")], excessmortalitypscoreinterpolated[CountryIndex(code, "SRB")]);
    write_country_pair("BurundiSeychellesCD.txt", firstdate, vaccinations[CountryIndex(code, "BDI")], vaccinations[CountryIndex(code, "SYC")], coviddeaths[CountryIndex(code, "BDI")], coviddeaths[CountryIndex(code, "SYC")]);
    write_country_pair("CambodiaVietnamCD.txt", firstdate, vaccinations[CountryIndex(code, "KHM")], vaccinations[CountryIndex(code, "VNM")], coviddeaths[CountryIndex(code, "KHM")], coviddeaths[CountryIndex(code, "VNM")]);
    write_country_pair("CanadaCubaEM.txt", firstdate, vaccinations[CountryIndex(code, "CAN")], vaccinations[CountryIndex(code, "CUB")], excessmortalitypscoreinterpolated[CountryIndex(code, "CAN")], excessmortalitypscoreinterpolated[CountryIndex(code, "CUB")]);
    write_country_pair("CanadaSaintVincentandtheGrenadinesEM.txt", firstdate, vaccinations[CountryIndex(code, "CAN")], vaccinations[CountryIndex(code, "VCT")], excessmortalitypscoreinterpolated[CountryIndex(code, "CAN")], excessmortalitypscoreinterpolated[CountryIndex(code, "VCT")]);
    write_country_pair("CapeVerdeTunisiaEM.txt", firstdate, vaccinations[CountryIndex(code, "CPV")], vaccinations[CountryIndex(code, "TUN")], excessmortalitypscoreinterpolated[CountryIndex(code, "CPV")], excessmortalitypscoreinterpolated[CountryIndex(code, "TUN")]);
    write_country_pair("CroatiaHungaryCD.txt", firstdate, vaccinations[CountryIndex(code, "HRV")], vaccinations[CountryIndex(code, "HUN")], coviddeaths[CountryIndex(code, "HRV")], coviddeaths[CountryIndex(code, "HUN")]);
    write_country_pair("CroatiaHungaryEM.txt", firstdate, vaccinations[CountryIndex(code, "HRV")], vaccinations[CountryIndex(code, "HUN")], excessmortalitypscoreinterpolated[CountryIndex(code, "HRV")], excessmortalitypscoreinterpolated[CountryIndex(code, "HUN")]);
    write_country_pair("CroatiaHungary_0_14EM.txt", firstdate, vaccinations[CountryIndex(code, "HRV")], vaccinations[CountryIndex(code, "HUN")], HRVexcessmortalitypscore_0_14interpolated, HUNexcessmortalitypscore_0_14interpolated);
    write_country_pair("CroatiaHungary_15_64EM.txt", firstdate, vaccinations[CountryIndex(code, "HRV")], vaccinations[CountryIndex(code, "HUN")], HRVexcessmortalitypscore_15_64interpolated, HUNexcessmortalitypscore_15_64interpolated);
    write_country_pair("CroatiaHungary_65_74EM.txt", firstdate, vaccinations[CountryIndex(code, "HRV")], vaccinations[CountryIndex(code, "HUN")], HRVexcessmortalitypscore_65_74interpolated, HUNexcessmortalitypscore_65_74interpolated);
    write_country_pair("CroatiaHungary_75_84EM.txt", firstdate, vaccinations[CountryIndex(code, "HRV")], vaccinations[CountryIndex(code, "HUN")], HRVexcessmortalitypscore_75_84interpolated, HUNexcessmortalitypscore_75_84interpolated);
    write_country_pair("CroatiaHungary_85pEM.txt", firstdate, vaccinations[CountryIndex(code, "HRV")], vaccinations[CountryIndex(code, "HUN")], HRVexcessmortalitypscore_85pinterpolated, HUNexcessmortalitypscore_85pinterpolated);
    write_country_pair("CubaJamaicaCD.txt", firstdate, vaccinations[CountryIndex(code, "CUB")], vaccinations[CountryIndex(code, "JAM")], coviddeaths[CountryIndex(code, "CUB")], coviddeaths[CountryIndex(code, "JAM")]);
    write_country_pair("CubaJamaicaEM.txt", firstdate, vaccinations[CountryIndex(code, "CUB")], vaccinations[CountryIndex(code, "JAM")], excessmortalitypscoreinterpolated[CountryIndex(code, "CUB")], excessmortalitypscoreinterpolated[CountryIndex(code, "JAM")]);
    write_country_pair("CubaSaintVincentandtheGrenadinesEM.txt", firstdate, vaccinations[CountryIndex(code, "CUB")], vaccinations[CountryIndex(code, "VCT")], excessmortalitypscoreinterpolated[CountryIndex(code, "CUB")], excessmortalitypscoreinterpolated[CountryIndex(code, "VCT")]);
    write_country_pair("DenmarkFinlandEM.txt", firstdate, vaccinations[CountryIndex(code, "DNK")], vaccinations[CountryIndex(code, "FIN")], excessmortalitypscoreinterpolated[CountryIndex(code, "DNK")], excessmortalitypscoreinterpolated[CountryIndex(code, "FIN")]);
    write_country_pair("DenmarkGermanyEM.txt", firstdate, vaccinations[CountryIndex(code, "DNK")], vaccinations[CountryIndex(code, "DEU")], excessmortalitypscoreinterpolated[CountryIndex(code, "DNK")], excessmortalitypscoreinterpolated[CountryIndex(code, "DEU")]);
    write_country_pair("DenmarkNorwayCD.txt", firstdate, vaccinations[CountryIndex(code, "DNK")], vaccinations[CountryIndex(code, "NOR")], coviddeaths[CountryIndex(code, "DNK")], coviddeaths[CountryIndex(code, "NOR")]);
    write_country_pair("DenmarkNorwayEM.txt", firstdate, vaccinations[CountryIndex(code, "DNK")], vaccinations[CountryIndex(code, "NOR")], excessmortalitypscoreinterpolated[CountryIndex(code, "DNK")], excessmortalitypscoreinterpolated[CountryIndex(code, "NOR")]);
    write_country_pair("DominicanRepublicHaitiCD.txt", firstdate, vaccinations[CountryIndex(code, "DOM")], vaccinations[CountryIndex(code, "HTI")], coviddeaths[CountryIndex(code, "DOM")], coviddeaths[CountryIndex(code, "HTI")]);
    write_country_pair("FaroeIslandsNorwayCD.txt", firstdate, vaccinations[CountryIndex(code, "FRO")], vaccinations[CountryIndex(code, "NOR")], coviddeaths[CountryIndex(code, "FRO")], coviddeaths[CountryIndex(code, "NOR")]);
    write_country_pair("FaroeIslandsNorwayEM.txt", firstdate, vaccinations[CountryIndex(code, "FRO")], vaccinations[CountryIndex(code, "NOR")], excessmortalitypscoreinterpolated[CountryIndex(code, "FRO")], excessmortalitypscoreinterpolated[CountryIndex(code, "NOR")]);
    write_country_pair("FrenchPolynesiaNewZealandEM.txt", firstdate, vaccinations[CountryIndex(code, "PYF")], vaccinations[CountryIndex(code, "NZL")], excessmortalitypscoreinterpolated[CountryIndex(code, "PYF")], excessmortalitypscoreinterpolated[CountryIndex(code, "NZL")]);
    write_country_pair("GibraltarIndiaCD.txt", firstdate, vaccinations[CountryIndex(code, "GIB")], vaccinations[CountryIndex(code, "IND")], coviddeaths[CountryIndex(code, "GIB")], coviddeaths[CountryIndex(code, "IND")]);
    write_country_pair("GreeceSerbiaEM.txt", firstdate, vaccinations[CountryIndex(code, "GRC")], vaccinations[CountryIndex(code, "SRB")], excessmortalitypscoreinterpolated[CountryIndex(code, "GRC")], excessmortalitypscoreinterpolated[CountryIndex(code, "SRB")]);
    write_country_pair("HongKongMacaoEM.txt", firstdate, vaccinations[CountryIndex(code, "HKG")], vaccinations[CountryIndex(code, "MAC")], excessmortalitypscoreinterpolated[CountryIndex(code, "HKG")], excessmortalitypscoreinterpolated[CountryIndex(code, "MAC")]);
    write_country_pair("HongKongPhilippinesCD.txt", firstdate, vaccinations[CountryIndex(code, "HKG")], vaccinations[CountryIndex(code, "PHL")], coviddeaths[CountryIndex(code, "HKG")], coviddeaths[CountryIndex(code, "PHL")]);
    write_country_pair("HongKongPhilippinesEM.txt", firstdate, vaccinations[CountryIndex(code, "HKG")], vaccinations[CountryIndex(code, "PHL")], excessmortalitypscoreinterpolated[CountryIndex(code, "HKG")], excessmortalitypscoreinterpolated[CountryIndex(code, "PHL")]);
    write_country_pair("HongKongSouthKoreaEM.txt", firstdate, vaccinations[CountryIndex(code, "HKG")], vaccinations[CountryIndex(code, "KOR")], excessmortalitypscoreinterpolated[CountryIndex(code, "HKG")], excessmortalitypscoreinterpolated[CountryIndex(code, "KOR")]);
    write_country_pair("HungaryRomaniaCD.txt", firstdate, vaccinations[CountryIndex(code, "HUN")], vaccinations[CountryIndex(code, "ROU")], coviddeaths[CountryIndex(code, "HUN")], coviddeaths[CountryIndex(code, "ROU")]);
    write_country_pair("HungaryRomaniaEM.txt", firstdate, vaccinations[CountryIndex(code, "HUN")], vaccinations[CountryIndex(code, "ROU")], excessmortalitypscoreinterpolated[CountryIndex(code, "HUN")], excessmortalitypscoreinterpolated[CountryIndex(code, "ROU")]);
    write_country_pair("IndonesiaPhilippinesCD.txt", firstdate, vaccinations[CountryIndex(code, "IDN")], vaccinations[CountryIndex(code, "PHL")], coviddeaths[CountryIndex(code, "IDN")], coviddeaths[CountryIndex(code, "PHL")]);
    write_country_pair("IranIraqCD.txt", firstdate, vaccinations[CountryIndex(code, "IRN")], vaccinations[CountryIndex(code, "IRQ")], coviddeaths[CountryIndex(code, "IRN")], coviddeaths[CountryIndex(code, "IRQ")]);
    write_country_pair("IraqOmanCD.txt", firstdate, vaccinations[CountryIndex(code, "IRQ")], vaccinations[CountryIndex(code, "OMN")], coviddeaths[CountryIndex(code, "IRQ")], coviddeaths[CountryIndex(code, "OMN")]);
    write_country_pair("JamaicaSaintVincentandtheGrenadinesCD.txt", firstdate, vaccinations[CountryIndex(code, "JAM")], vaccinations[CountryIndex(code, "VCT")], coviddeaths[CountryIndex(code, "JAM")], coviddeaths[CountryIndex(code, "VCT")]);
    write_country_pair("JamaicaSaintVincentandtheGrenadinesEM.txt", firstdate, vaccinations[CountryIndex(code, "JAM")], vaccinations[CountryIndex(code, "VCT")], excessmortalitypscoreinterpolated[CountryIndex(code, "JAM")], excessmortalitypscoreinterpolated[CountryIndex(code, "VCT")]);
    write_country_pair("JapanMalaysiaEM.txt", firstdate, vaccinations[CountryIndex(code, "JPN")], vaccinations[CountryIndex(code, "MYS")], excessmortalitypscoreinterpolated[CountryIndex(code, "JPN")], excessmortalitypscoreinterpolated[CountryIndex(code, "MYS")]);
    write_country_pair("JapanPhilippinesEM.txt", firstdate, vaccinations[CountryIndex(code, "JPN")], vaccinations[CountryIndex(code, "PHL")], excessmortalitypscoreinterpolated[CountryIndex(code, "JPN")], excessmortalitypscoreinterpolated[CountryIndex(code, "PHL")]);
    write_country_pair("JapanSouthKoreaEM.txt", firstdate, vaccinations[CountryIndex(code, "JPN")], vaccinations[CountryIndex(code, "KOR")], excessmortalitypscoreinterpolated[CountryIndex(code, "JPN")], excessmortalitypscoreinterpolated[CountryIndex(code, "KOR")]);
    write_country_pair("KyrgyzstanUzbekistanCD.txt", firstdate, vaccinations[CountryIndex(code, "KGZ")], vaccinations[CountryIndex(code, "UZB")], coviddeaths[CountryIndex(code, "KGZ")], coviddeaths[CountryIndex(code, "UZB")]);
    write_country_pair("KyrgyzstanUzbekistanEM.txt", firstdate, vaccinations[CountryIndex(code, "KGZ")], vaccinations[CountryIndex(code, "UZB")], excessmortalitypscoreinterpolated[CountryIndex(code, "KGZ")], excessmortalitypscoreinterpolated[CountryIndex(code, "UZB")]);
    write_country_pair("LatviaUkraineCD.txt", firstdate, vaccinations[CountryIndex(code, "LVA")], vaccinations[CountryIndex(code, "UKR")], coviddeaths[CountryIndex(code, "LVA")], coviddeaths[CountryIndex(code, "UKR")]);
    write_country_pair("LatviaUkraineEM.txt", firstdate, vaccinations[CountryIndex(code, "LVA")], vaccinations[CountryIndex(code, "UKR")], excessmortalitypscoreinterpolated[CountryIndex(code, "LVA")], excessmortalitypscoreinterpolated[CountryIndex(code, "UKR")]);
    write_country_pair("LebanonPalestineEM.txt", firstdate, vaccinations[CountryIndex(code, "LBN")], vaccinations[CountryIndex(code, "PSE")], excessmortalitypscoreinterpolated[CountryIndex(code, "LBN")], excessmortalitypscoreinterpolated[CountryIndex(code, "PSE")]);
    write_country_pair("MalaysiaMongoliaEM.txt", firstdate, vaccinations[CountryIndex(code, "MYS")], vaccinations[CountryIndex(code, "MNG")], excessmortalitypscoreinterpolated[CountryIndex(code, "MYS")], excessmortalitypscoreinterpolated[CountryIndex(code, "MNG")]);
    write_country_pair("MalaysiaTaiwanEM.txt", firstdate, vaccinations[CountryIndex(code, "MYS")], vaccinations[CountryIndex(code, "TWN")], excessmortalitypscoreinterpolated[CountryIndex(code, "MYS")], excessmortalitypscoreinterpolated[CountryIndex(code, "TWN")]);
    write_country_pair("MalaysiaThailandCD.txt", firstdate, vaccinations[CountryIndex(code, "MYS")], vaccinations[CountryIndex(code, "THA")], coviddeaths[CountryIndex(code, "MYS")], coviddeaths[CountryIndex(code, "THA")]);
    write_country_pair("MalaysiaThailandEM.txt", firstdate, vaccinations[CountryIndex(code, "MYS")], vaccinations[CountryIndex(code, "THA")], excessmortalitypscoreinterpolated[CountryIndex(code, "MYS")], excessmortalitypscoreinterpolated[CountryIndex(code, "THA")]);
    write_country_pair("MauritiusSeychellesCD.txt", firstdate, vaccinations[CountryIndex(code, "MUS")], vaccinations[CountryIndex(code, "SYC")], coviddeaths[CountryIndex(code, "MUS")], coviddeaths[CountryIndex(code, "SYC")]);
    write_country_pair("MauritiusSeychellesEM.txt", firstdate, vaccinations[CountryIndex(code, "MUS")], vaccinations[CountryIndex(code, "SYC")], excessmortalitypscoreinterpolated[CountryIndex(code, "MUS")], excessmortalitypscoreinterpolated[CountryIndex(code, "SYC")]);
    write_country_pair("MongoliaThailandCD.txt", firstdate, vaccinations[CountryIndex(code, "MNG")], vaccinations[CountryIndex(code, "THA")], coviddeaths[CountryIndex(code, "MNG")], coviddeaths[CountryIndex(code, "THA")]);
    write_country_pair("MongoliaThailandEM.txt", firstdate, vaccinations[CountryIndex(code, "MNG")], vaccinations[CountryIndex(code, "THA")], excessmortalitypscoreinterpolated[CountryIndex(code, "MNG")], excessmortalitypscoreinterpolated[CountryIndex(code, "THA")]);
    write_country_pair("PapuaNewGuineaSolomonIslandsCD.txt", firstdate, vaccinations[CountryIndex(code, "PNG")], vaccinations[CountryIndex(code, "SLB")], coviddeaths[CountryIndex(code, "PNG")], coviddeaths[CountryIndex(code, "SLB")]);
    write_country_pair("PhilippinesSouthKoreaEM.txt", firstdate, vaccinations[CountryIndex(code, "PHL")], vaccinations[CountryIndex(code, "KOR")], excessmortalitypscoreinterpolated[CountryIndex(code, "PHL")], excessmortalitypscoreinterpolated[CountryIndex(code, "KOR")]);
    write_country_pair("RomaniaRussiaCD.txt", firstdate, vaccinations[CountryIndex(code, "ROU")], vaccinations[CountryIndex(code, "RUS")], coviddeaths[CountryIndex(code, "ROU")], coviddeaths[CountryIndex(code, "RUS")]);
    write_country_pair("RomaniaRussiaEM.txt", firstdate, vaccinations[CountryIndex(code, "ROU")], vaccinations[CountryIndex(code, "RUS")], excessmortalitypscoreinterpolated[CountryIndex(code, "ROU")], excessmortalitypscoreinterpolated[CountryIndex(code, "RUS")]);
    write_country_pair("RwandaUgandaCD.txt", firstdate, vaccinations[CountryIndex(code, "RWA")], vaccinations[CountryIndex(code, "UGA")], coviddeaths[CountryIndex(code, "RWA")], coviddeaths[CountryIndex(code, "UGA")]);
    write_country_pair("SeychellesTanzaniaCD.txt", firstdate, vaccinations[CountryIndex(code, "SYC")], vaccinations[CountryIndex(code, "TZA")], coviddeaths[CountryIndex(code, "SYC")], coviddeaths[CountryIndex(code, "TZA")]);
    write_country_pair("SingaporeThailandCD.txt", firstdate, vaccinations[CountryIndex(code, "SGP")], vaccinations[CountryIndex(code, "THA")], coviddeaths[CountryIndex(code, "SGP")], coviddeaths[CountryIndex(code, "THA")]);
    write_country_pair("SingaporeThailandEM.txt", firstdate, vaccinations[CountryIndex(code, "SGP")], vaccinations[CountryIndex(code, "THA")], excessmortalitypscoreinterpolated[CountryIndex(code, "SGP")], excessmortalitypscoreinterpolated[CountryIndex(code, "THA")]);
    write_country_pair("SWEGBR.txt", firstdate, lockdown[CountryIndex(code, "SWE")], lockdown[CountryIndex(code, "GBR")], excessmortalitypscoreinterpolated[CountryIndex(code, "SWE")], excessmortalitypscoreinterpolated[CountryIndex(code, "GBR")]);
    write_country_pair("ThailandVietnamCD.txt", firstdate, vaccinations[CountryIndex(code, "THA")], vaccinations[CountryIndex(code, "VNM")], coviddeaths[CountryIndex(code, "THA")], coviddeaths[CountryIndex(code, "VNM")]);
    write_country_pair("ZambiaZimbabweCD.txt", firstdate, vaccinations[CountryIndex(code, "ZMB")], vaccinations[CountryIndex(code, "ZWE")], coviddeaths[CountryIndex(code, "ZMB")], coviddeaths[CountryIndex(code, "ZWE")]);

#ifndef OWID_SKIP_DTW
	std::string vaxdiff, cddiff, emdiff;

	for (int c1 = 0; c1 < numentities; c1++)
		for (int c2 = 0; c2 < numentities; c2++) {
			// Only compare distinct countries on the same known continent.
            if (c1 >= c2 || continent[c1].empty() || continent[c1] != continent[c2]) continue;
            // Legacy DTW coordinates combine days and outcome values. Distances are
            // arbitrary-unit matching scores, not physical rates or causal effects.

			// compare vaccinations
			std::vector<std::vector<double> > v1(numdates, std::vector<double>(2));
			std::vector<std::vector<double> > v2(numdates, std::vector<double>(2));
			for (int d1 = 0; d1 < numdates; d1++) {
				v1[d1][0] = d1;
				v1[d1][1] = vaccinations[c1][d1];
				v2[d1][0] = d1;
				v2[d1][1] = vaccinations[c2][d1];
			}
			for (unsigned int i = 0; i < v1.size(); ) {
				if (std::isnan(v1[i][1]) || std::isnan(v2[i][1])) {
					v1.erase(v1.begin() + i);
					v2.erase(v2.begin() + i);
				}
				else
					++i;
			}

			// compare covid deaths
			std::vector<std::vector<double> > cd1(numprevaxdates, std::vector<double>(2));
			std::vector<std::vector<double> > cd2(numprevaxdates, std::vector<double>(2));
			for (unsigned int d1 = 0; d1 < numprevaxdates; d1++) {
				cd1[d1][0] = d1;
				cd1[d1][1] = coviddeaths[c1][d1];
				cd2[d1][0] = d1;
				cd2[d1][1] = coviddeaths[c2][d1];
			}
			for (unsigned int i = 0; i < cd1.size(); ) {
				if (std::isnan(cd1[i][1]) || std::isnan(cd2[i][1])) {
					cd1.erase(cd1.begin() + i);
					cd2.erase(cd2.begin() + i);
				}
				else
					++i;
			}

			// compare excess deaths
			std::vector<std::vector<double> > em1(numprevaxdates, std::vector<double>(2));
			std::vector<std::vector<double> > em2(numprevaxdates, std::vector<double>(2));
			for (int d1 = 0; d1 < numprevaxdates; d1++) {
				em1[d1][0] = d1;
				em1[d1][1] = excessmortalitypscoreprevax[c1][d1];
				em2[d1][0] = d1;
				em2[d1][1] = excessmortalitypscoreprevax[c2][d1];
			}
			for (unsigned int i = 0; i < em1.size(); ) {
				if (std::isnan(em1[i][1]) || std::isnan(em2[i][1])) {
					em1.erase(em1.begin() + i);
					em2.erase(em2.begin() + i);
				}
				else
					++i;
			}

			if ((continent[c1].compare(continent[c2]) == 0) && (c1 < c2)){
				if (v1.size() > 0)
					vaxdiff = tostring(DTW::dtw_distance_only(v1, v2, 2));
				else
					vaxdiff = "nan";
				if (cd1.size() > 0)
					cddiff = tostring(DTW::dtw_distance_only(cd1, cd2, 2));
				else
					cddiff = "nan";
				if (em1.size() > 0)
					emdiff = tostring(DTW::dtw_distance_only(em1, em2, 2));
				else
					emdiff = "nan";
				display.push_back(continent[c1] + "\t" + entity[c1] + "\t" + entity[c2] + "\t" + vaxdiff + "\t" + cddiff + "\t" + emdiff);
			}
		}
	std::cout << ".";

	std::sort(display.begin(), display.end());
	outputfile << "Continent" <<  "\t" << "Country1" << "\t" << "Country2" << "\t" << "Vaccination DTW distance (mixed-coordinate arbitrary units)" << "\t" << "COVID-19 DTW distance (mixed-coordinate arbitrary units)" << "\t" << "Excess mortality DTW distance (mixed-coordinate arbitrary units)" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;

#else
    outputfile << "DTW distance calculations were skipped (compiled with OWID_SKIP_DTW).\n";
    std::cout << "DTW distance calculations skipped; DTW.hpp was not compiled.\n";
#endif
	std::cout << std::endl;
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;
	double mins = std::floor(diff.count() / 60);
	double secs = diff.count() - mins*60;
	//std::cout << "Elapsed time: " << mins << " minutes " << secs << " seconds" << std::endl;
	std::cout << "Finished!" << std::endl;
	std::cout << "5) Copy this program's output files to the directory containing SIR.py." << std::endl;
	std::cout << "6) Run SIR.py." << std::endl;
	if (!use_clusters) std::cin.get();
	return 0;
} catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
}
