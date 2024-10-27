// The Effectiveness of Lockdowns, Face Masks and Vaccination Programmes Vis-à-Vis Mitigating COVID-19
// Martin Sewell
// martin.sewell@cantab.net
// 22 October 2024

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
// used for coviddeaths[] and cumcoviddeaths[]
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
#include <DTW.hpp>

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
	long double sum = accumulate(v.begin(), v.end(), 0.0);
	long double mean = sum / (long double)n;
	return mean;
}


int CountryIndex(std::vector<std::string> list, std::string item)
{
	if (!item.empty()) {
		auto it = find(list.begin(), list.end(), item);
		if (it != list.end()) {
			int index = it - list.begin();
			return index;
		}
		else {
			std::cout << item << " not found" << " ";
			return -1;
		}
	}
	else {
		std::cout << "CountryIndex error";
		return -1;
	}
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
		if (!isnan(v[d])) {
			x.push_back(double(d));
			y.push_back(v[d]);
		}
	}
	int first = -1;
	int last = -1;
	bool firstcase = true;
	for (unsigned int d = 0; d < interpolated.size(); d++)
		if (!isnan(v[d])) {
			if (firstcase == true) {
				first = d;
				firstcase = false;
			}
			last = d;
		}
	if (x.size() > 0) {
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




int doclustering(int c, std::vector<int> clustcd, std::vector<int> clustem, std::vector < std::vector<double> > vaccinations, std::vector<double> cumvax, std::vector < std::vector<double> > coviddeaths, std::vector < std::vector<double> > excessmortalitypscoreinterpolated) {


	std::vector<int> vaxlowcd0;
	std::vector<int> vaxmedcd0;
	std::vector<int> vaxhighcd0;

	std::vector<int> vaxlowcd1;
	std::vector<int> vaxmedcd1;
	std::vector<int> vaxhighcd1;

	std::vector<int> vaxlowcd2;
	std::vector<int> vaxmedcd2;
	std::vector<int> vaxhighcd2;

	std::vector<int> vaxlowcd3;
	std::vector<int> vaxmedcd3;
	std::vector<int> vaxhighcd3;

	std::vector<int> vaxlowcd4;
	std::vector<int> vaxmedcd4;
	std::vector<int> vaxhighcd4;

	std::vector<int> vaxlowcd5;
	std::vector<int> vaxmedcd5;
	std::vector<int> vaxhighcd5;

	std::vector<int> vaxlowcd6;
	std::vector<int> vaxmedcd6;
	std::vector<int> vaxhighcd6;

	std::vector<int> vaxlowcd7;
	std::vector<int> vaxmedcd7;
	std::vector<int> vaxhighcd7;

	std::vector<int> vaxlowcd8;
	std::vector<int> vaxmedcd8;
	std::vector<int> vaxhighcd8;

	std::vector<int> vaxlowcd9;
	std::vector<int> vaxmedcd9;
	std::vector<int> vaxhighcd9;

	std::vector<int> vaxlowcd10;
	std::vector<int> vaxmedcd10;
	std::vector<int> vaxhighcd10;

	std::vector<int> vaxlowcd11;
	std::vector<int> vaxmedcd11;
	std::vector<int> vaxhighcd11;

	std::vector<int> vaxlowcd12;
	std::vector<int> vaxmedcd12;
	std::vector<int> vaxhighcd12;

	std::vector<int> vaxlowcd13;
	std::vector<int> vaxmedcd13;
	std::vector<int> vaxhighcd13;

	std::vector<int> vaxlowcd14;
	std::vector<int> vaxmedcd14;
	std::vector<int> vaxhighcd14;

	std::vector<int> vaxlowcd15;
	std::vector<int> vaxmedcd15;
	std::vector<int> vaxhighcd15;

	std::vector<int> vaxlowcd16;
	std::vector<int> vaxmedcd16;
	std::vector<int> vaxhighcd16;

	std::vector<int> vaxlowcd17;
	std::vector<int> vaxmedcd17;
	std::vector<int> vaxhighcd17;

	std::vector<int> vaxlowcd18;
	std::vector<int> vaxmedcd18;
	std::vector<int> vaxhighcd18;

	std::vector<int> vaxlowcd19;
	std::vector<int> vaxmedcd19;
	std::vector<int> vaxhighcd19;

	std::vector<int> vaxlowem0;
	std::vector<int> vaxmedem0;
	std::vector<int> vaxhighem0;

	std::vector<int> vaxlowem1;
	std::vector<int> vaxmedem1;
	std::vector<int> vaxhighem1;

	std::vector<int> vaxlowem2;
	std::vector<int> vaxmedem2;
	std::vector<int> vaxhighem2;

	std::vector<int> vaxlowem3;
	std::vector<int> vaxmedem3;
	std::vector<int> vaxhighem3;

	std::vector<int> vaxlowem4;
	std::vector<int> vaxmedem4;
	std::vector<int> vaxhighem4;

	std::vector<int> vaxlowem5;
	std::vector<int> vaxmedem5;
	std::vector<int> vaxhighem5;

	std::vector<int> vaxlowem6;
	std::vector<int> vaxmedem6;
	std::vector<int> vaxhighem6;

	std::vector<int> vaxlowem7;
	std::vector<int> vaxmedem7;
	std::vector<int> vaxhighem7;

	std::vector<int> vaxlowem8;
	std::vector<int> vaxmedem8;
	std::vector<int> vaxhighem8;

	std::vector<int> vaxlowem9;
	std::vector<int> vaxmedem9;
	std::vector<int> vaxhighem9;

	std::vector<int> vaxlowem10;
	std::vector<int> vaxmedem10;
	std::vector<int> vaxhighem10;

	std::vector<int> vaxlowem11;
	std::vector<int> vaxmedem11;
	std::vector<int> vaxhighem11;

	std::vector<int> vaxlowem12;
	std::vector<int> vaxmedem12;
	std::vector<int> vaxhighem12;

	std::vector<int> vaxlowem13;
	std::vector<int> vaxmedem13;
	std::vector<int> vaxhighem13;

	std::vector<int> vaxlowem14;
	std::vector<int> vaxmedem14;
	std::vector<int> vaxhighem14;

	std::vector<int> vaxlowem15;
	std::vector<int> vaxmedem15;
	std::vector<int> vaxhighem15;

	std::vector<int> vaxlowem16;
	std::vector<int> vaxmedem16;
	std::vector<int> vaxhighem16;

	std::vector<int> vaxlowem17;
	std::vector<int> vaxmedem17;
	std::vector<int> vaxhighem17;

	std::vector<int> vaxlowem18;
	std::vector<int> vaxmedem18;
	std::vector<int> vaxhighem18;

	std::vector<int> vaxlowem19;
	std::vector<int> vaxmedem19;
	std::vector<int> vaxhighem19;


	// clustcd[c] and clustem[c] only include clustering countries
	int ci = 0;
	for (unsigned int i = 0; i < numentities; i++) {
		if (std::find(vaxlow.begin(), vaxlow.end(), i) != vaxlow.end()) {
			if (clustcd[ci] == 0)
				vaxlowcd0.push_back(i);
			if (clustcd[ci] == 1)
				vaxlowcd1.push_back(i);
			if (clustcd[ci] == 2)
				vaxlowcd2.push_back(i);
			if (clustcd[ci] == 3)
				vaxlowcd3.push_back(i);
			if (clustcd[ci] == 4)
				vaxlowcd4.push_back(i);
			if (clustcd[ci] == 5)
				vaxlowcd5.push_back(i);
			if (clustcd[ci] == 6)
				vaxlowcd6.push_back(i);
			if (clustcd[ci] == 7)
				vaxlowcd7.push_back(i);
			if (clustcd[ci] == 8)
				vaxlowcd8.push_back(i);
			if (clustcd[ci] == 9)
				vaxlowcd9.push_back(i);
			if (clustcd[ci] == 10)
				vaxlowcd10.push_back(i);
			if (clustcd[ci] == 11)
				vaxlowcd11.push_back(i);
			if (clustcd[ci] == 12)
				vaxlowcd12.push_back(i);
			if (clustcd[ci] == 13)
				vaxlowcd13.push_back(i);
			if (clustcd[ci] == 14)
				vaxlowcd14.push_back(i);
			if (clustcd[ci] == 15)
				vaxlowcd15.push_back(i);
			if (clustcd[ci] == 16)
				vaxlowcd16.push_back(i);
			if (clustcd[ci] == 17)
				vaxlowcd17.push_back(i);
			if (clustcd[ci] == 18)
				vaxlowcd18.push_back(i);
			if (clustcd[ci] == 19)
				vaxlowcd19.push_back(i);

			if (clustem[ci] == 0)
				vaxlowem0.push_back(i);
			if (clustem[ci] == 1)
				vaxlowem1.push_back(i);
			if (clustem[ci] == 2)
				vaxlowem2.push_back(i);
			if (clustem[ci] == 3)
				vaxlowem3.push_back(i);
			if (clustem[ci] == 4)
				vaxlowem4.push_back(i);
			if (clustem[ci] == 5)
				vaxlowem5.push_back(i);
			if (clustem[ci] == 6)
				vaxlowem6.push_back(i);
			if (clustem[ci] == 7)
				vaxlowem7.push_back(i);
			if (clustem[ci] == 8)
				vaxlowem8.push_back(i);
			if (clustem[ci] == 9)
				vaxlowem9.push_back(i);
			if (clustem[ci] == 10)
				vaxlowem10.push_back(i);
			if (clustem[ci] == 11)
				vaxlowem11.push_back(i);
			if (clustem[ci] == 12)
				vaxlowem12.push_back(i);
			if (clustem[ci] == 13)
				vaxlowem13.push_back(i);
			if (clustem[ci] == 14)
				vaxlowem14.push_back(i);
			if (clustem[ci] == 15)
				vaxlowem15.push_back(i);
			if (clustem[ci] == 16)
				vaxlowem16.push_back(i);
			if (clustem[ci] == 17)
				vaxlowem17.push_back(i);
			if (clustem[ci] == 18)
				vaxlowem18.push_back(i);
			if (clustem[ci] == 19)
				vaxlowem19.push_back(i);
			ci++;
		}
		if (std::find(vaxmed.begin(), vaxmed.end(), i) != vaxmed.end()) {
			if (clustcd[ci] == 0)
				vaxmedcd0.push_back(i);
			if (clustcd[ci] == 1)
				vaxmedcd1.push_back(i);
			if (clustcd[ci] == 2)
				vaxmedcd2.push_back(i);
			if (clustcd[ci] == 3)
				vaxmedcd3.push_back(i);
			if (clustcd[ci] == 4)
				vaxmedcd4.push_back(i);
			if (clustcd[ci] == 5)
				vaxmedcd5.push_back(i);
			if (clustcd[ci] == 6)
				vaxmedcd6.push_back(i);
			if (clustcd[ci] == 7)
				vaxmedcd7.push_back(i);
			if (clustcd[ci] == 8)
				vaxmedcd8.push_back(i);
			if (clustcd[ci] == 9)
				vaxmedcd9.push_back(i);
			if (clustcd[ci] == 10)
				vaxmedcd10.push_back(i);
			if (clustcd[ci] == 11)
				vaxmedcd11.push_back(i);
			if (clustcd[ci] == 12)
				vaxmedcd12.push_back(i);
			if (clustcd[ci] == 13)
				vaxmedcd13.push_back(i);
			if (clustcd[ci] == 14)
				vaxmedcd14.push_back(i);
			if (clustcd[ci] == 15)
				vaxmedcd15.push_back(i);
			if (clustcd[ci] == 16)
				vaxmedcd16.push_back(i);
			if (clustcd[ci] == 17)
				vaxmedcd17.push_back(i);
			if (clustcd[ci] == 18)
				vaxmedcd18.push_back(i);
			if (clustcd[ci] == 19)
				vaxmedcd19.push_back(i);


			if (clustem[ci] == 0)
				vaxmedem0.push_back(i);
			if (clustem[ci] == 1)
				vaxmedem1.push_back(i);
			if (clustem[ci] == 2)
				vaxmedem2.push_back(i);
			if (clustem[ci] == 3)
				vaxmedem3.push_back(i);
			if (clustem[ci] == 4)
				vaxmedem4.push_back(i);
			if (clustem[ci] == 5)
				vaxmedem5.push_back(i);
			if (clustem[ci] == 6)
				vaxmedem6.push_back(i);
			if (clustem[ci] == 7)
				vaxmedem7.push_back(i);
			if (clustem[ci] == 8)
				vaxmedem8.push_back(i);
			if (clustem[ci] == 9)
				vaxmedem9.push_back(i);
			if (clustem[ci] == 10)
				vaxmedem10.push_back(i);
			if (clustem[ci] == 11)
				vaxmedem11.push_back(i);
			if (clustem[ci] == 12)
				vaxmedem12.push_back(i);
			if (clustem[ci] == 13)
				vaxmedem13.push_back(i);
			if (clustem[ci] == 14)
				vaxmedem14.push_back(i);
			if (clustem[ci] == 15)
				vaxmedem15.push_back(i);
			if (clustem[ci] == 16)
				vaxmedem16.push_back(i);
			if (clustem[ci] == 17)
				vaxmedem17.push_back(i);
			if (clustem[ci] == 18)
				vaxmedem18.push_back(i);
			if (clustem[ci] == 19)
				vaxmedem19.push_back(i);

			ci++;
		}

		if (std::find(vaxhigh.begin(), vaxhigh.end(), i) != vaxhigh.end()) {
			if (clustcd[ci] == 0)
				vaxhighcd0.push_back(i);
			if (clustcd[ci] == 1)
				vaxhighcd1.push_back(i);
			if (clustcd[ci] == 2)
				vaxhighcd2.push_back(i);
			if (clustcd[ci] == 3)
				vaxhighcd3.push_back(i);
			if (clustcd[ci] == 4)
				vaxhighcd4.push_back(i);
			if (clustcd[ci] == 5)
				vaxhighcd5.push_back(i);
			if (clustcd[ci] == 6)
				vaxhighcd6.push_back(i);
			if (clustcd[ci] == 7)
				vaxhighcd7.push_back(i);
			if (clustcd[ci] == 8)
				vaxhighcd8.push_back(i);
			if (clustcd[ci] == 9)
				vaxhighcd9.push_back(i);
			if (clustcd[ci] == 10)
				vaxhighcd10.push_back(i);
			if (clustcd[ci] == 11)
				vaxhighcd11.push_back(i);
			if (clustcd[ci] == 12)
				vaxhighcd12.push_back(i);
			if (clustcd[ci] == 13)
				vaxhighcd13.push_back(i);
			if (clustcd[ci] == 14)
				vaxhighcd14.push_back(i);
			if (clustcd[ci] == 15)
				vaxhighcd15.push_back(i);
			if (clustcd[ci] == 16)
				vaxhighcd16.push_back(i);
			if (clustcd[ci] == 17)
				vaxhighcd17.push_back(i);
			if (clustcd[ci] == 18)
				vaxhighcd18.push_back(i);
			if (clustcd[ci] == 19)
				vaxhighcd19.push_back(i);


			if (clustem[ci] == 0)
				vaxhighem0.push_back(i);
			if (clustem[ci] == 1)
				vaxhighem1.push_back(i);
			if (clustem[ci] == 2)
				vaxhighem2.push_back(i);
			if (clustem[ci] == 3)
				vaxhighem3.push_back(i);
			if (clustem[ci] == 4)
				vaxhighem4.push_back(i);
			if (clustem[ci] == 5)
				vaxhighem5.push_back(i);
			if (clustem[ci] == 6)
				vaxhighem6.push_back(i);
			if (clustem[ci] == 7)
				vaxhighem7.push_back(i);
			if (clustem[ci] == 8)
				vaxhighem8.push_back(i);
			if (clustem[ci] == 9)
				vaxhighem9.push_back(i);
			if (clustem[ci] == 10)
				vaxhighem10.push_back(i);
			if (clustem[ci] == 11)
				vaxhighem11.push_back(i);
			if (clustem[ci] == 12)
				vaxhighem12.push_back(i);
			if (clustem[ci] == 13)
				vaxhighem13.push_back(i);
			if (clustem[ci] == 14)
				vaxhighem14.push_back(i);
			if (clustem[ci] == 15)
				vaxhighem15.push_back(i);
			if (clustem[ci] == 16)
				vaxhighem16.push_back(i);
			if (clustem[ci] == 17)
				vaxhighem17.push_back(i);
			if (clustem[ci] == 18)
				vaxhighem18.push_back(i);
			if (clustem[ci] == 19)
				vaxhighem19.push_back(i);
			ci++;
		}
		
	}


	std::string s = std::to_string(c);


	std::string clustcd0_vaxlow_filename;
	clustcd0_vaxlow_filename = s + "_clustcd0_vaxlow.txt";
	std::ofstream clustcd0_vaxlow_file;
	clustcd0_vaxlow_file.open(clustcd0_vaxlow_filename);

	std::string clustcd0_vaxlow_vx_filename;
	clustcd0_vaxlow_vx_filename = s + "_clustcd0_vaxlow_vx.txt";
	std::ofstream clustcd0_vaxlow_vx_file;
	clustcd0_vaxlow_vx_file.open(clustcd0_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd0.size(); i++) {
			clustcd0_vaxlow_vx_file << vaccinations[vaxlowcd0[i]][d] << "\t";
			clustcd0_vaxlow_file << coviddeaths[vaxlowcd0[i]][d] << "\t";
		}
		clustcd0_vaxlow_vx_file << std::endl;
		clustcd0_vaxlow_file << std::endl;
	}

	std::string clustcd0_vaxmed_filename;
	clustcd0_vaxmed_filename = s + "_clustcd0_vaxmed.txt";
	std::ofstream clustcd0_vaxmed_file;
	clustcd0_vaxmed_file.open(clustcd0_vaxmed_filename);

	std::string clustcd0_vaxmed_vx_filename;
	clustcd0_vaxmed_vx_filename = s + "_clustcd0_vaxmed_vx.txt";
	std::ofstream clustcd0_vaxmed_vx_file;
	clustcd0_vaxmed_vx_file.open(clustcd0_vaxmed_vx_filename);


	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd0.size(); i++) {
			clustcd0_vaxmed_vx_file << vaccinations[vaxmedcd0[i]][d] << "\t";
			clustcd0_vaxmed_file << coviddeaths[vaxmedcd0[i]][d] << "\t";
		}
		clustcd0_vaxmed_vx_file << std::endl;
		clustcd0_vaxmed_file << std::endl;
	}

	std::string clustcd0_vaxhigh_filename;
	clustcd0_vaxhigh_filename = s + "_clustcd0_vaxhigh.txt";
	std::ofstream clustcd0_vaxhigh_file;
	clustcd0_vaxhigh_file.open(clustcd0_vaxhigh_filename);

	std::string clustcd0_vaxhigh_vx_filename;
	clustcd0_vaxhigh_vx_filename = s + "_clustcd0_vaxhigh_vx.txt";
	std::ofstream clustcd0_vaxhigh_vx_file;
	clustcd0_vaxhigh_vx_file.open(clustcd0_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd0.size(); i++) {
			clustcd0_vaxhigh_vx_file << vaccinations[vaxhighcd0[i]][d] << "\t";
			clustcd0_vaxhigh_file << coviddeaths[vaxhighcd0[i]][d] << "\t";
		}
		clustcd0_vaxhigh_vx_file << std::endl;
		clustcd0_vaxhigh_file << std::endl;
	}

	std::string clustcd1_vaxlow_filename;
	clustcd1_vaxlow_filename = s + "_clustcd1_vaxlow.txt";
	std::ofstream clustcd1_vaxlow_file;
	clustcd1_vaxlow_file.open(clustcd1_vaxlow_filename);

	std::string clustcd1_vaxlow_vx_filename;
	clustcd1_vaxlow_vx_filename = s + "_clustcd1_vaxlow_vx.txt";
	std::ofstream clustcd1_vaxlow_vx_file;
	clustcd1_vaxlow_vx_file.open(clustcd1_vaxlow_vx_filename);

	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd1.size(); i++) {
			clustcd1_vaxlow_vx_file << vaccinations[vaxlowcd1[i]][d] << "\t";
			clustcd1_vaxlow_file << coviddeaths[vaxlowcd1[i]][d] << "\t";
		}
		clustcd1_vaxlow_file << std::endl;
		clustcd1_vaxlow_vx_file << std::endl;
	}

	std::string clustcd1_vaxmed_filename;
	clustcd1_vaxmed_filename = s + "_clustcd1_vaxmed.txt";
	std::ofstream clustcd1_vaxmed_file;
	clustcd1_vaxmed_file.open(clustcd1_vaxmed_filename);

	std::string clustcd1_vaxmed_vx_filename;
	clustcd1_vaxmed_vx_filename = s + "_clustcd1_vaxmed_vx.txt";
	std::ofstream clustcd1_vaxmed_vx_file;
	clustcd1_vaxmed_vx_file.open(clustcd1_vaxmed_vx_filename);

	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd1.size(); i++) {
			clustcd1_vaxmed_vx_file << vaccinations[vaxmedcd1[i]][d] << "\t";
			clustcd1_vaxmed_file << coviddeaths[vaxmedcd1[i]][d] << "\t";
		}
		clustcd1_vaxmed_vx_file << std::endl;
		clustcd1_vaxmed_file << std::endl;
	}

	std::string clustcd1_vaxhigh_filename;
	clustcd1_vaxhigh_filename = s + "_clustcd1_vaxhigh.txt";
	std::ofstream clustcd1_vaxhigh_file;
	clustcd1_vaxhigh_file.open(clustcd1_vaxhigh_filename);

	std::string clustcd1_vaxhigh_vx_filename;
	clustcd1_vaxhigh_vx_filename = s + "_clustcd1_vaxhigh_vx.txt";
	std::ofstream clustcd1_vaxhigh_vx_file;
	clustcd1_vaxhigh_vx_file.open(clustcd1_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd1.size(); i++) {
			clustcd1_vaxhigh_vx_file << vaccinations[vaxhighcd1[i]][d] << "\t";
			clustcd1_vaxhigh_file << coviddeaths[vaxhighcd1[i]][d] << "\t";
		}
		clustcd1_vaxhigh_vx_file << std::endl;
		clustcd1_vaxhigh_file << std::endl;
	}

	std::string clustcd2_vaxlow_filename;
	clustcd2_vaxlow_filename = s + "_clustcd2_vaxlow.txt";
	std::ofstream clustcd2_vaxlow_file;
	clustcd2_vaxlow_file.open(clustcd2_vaxlow_filename);
	std::string clustcd2_vaxlow_vx_filename;
	clustcd2_vaxlow_vx_filename = s + "_clustcd2_vaxlow_vx.txt";
	std::ofstream clustcd2_vaxlow_vx_file;
	clustcd2_vaxlow_vx_file.open(clustcd2_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd2.size(); i++) {
			clustcd2_vaxlow_vx_file << vaccinations[vaxlowcd2[i]][d] << "\t";
			clustcd2_vaxlow_file << coviddeaths[vaxlowcd2[i]][d] << "\t";
		}
		clustcd2_vaxlow_vx_file << std::endl;
		clustcd2_vaxlow_file << std::endl;
	}
	std::string clustcd2_vaxmed_filename;
	clustcd2_vaxmed_filename = s + "_clustcd2_vaxmed.txt";
	std::ofstream clustcd2_vaxmed_file;
	clustcd2_vaxmed_file.open(clustcd2_vaxmed_filename);
	std::string clustcd2_vaxmed_vx_filename;
	clustcd2_vaxmed_vx_filename = s + "_clustcd2_vaxmed_vx.txt";
	std::ofstream clustcd2_vaxmed_vx_file;
	clustcd2_vaxmed_vx_file.open(clustcd2_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd2.size(); i++) {
			clustcd2_vaxmed_vx_file << vaccinations[vaxmedcd2[i]][d] << "\t";
			clustcd2_vaxmed_file << coviddeaths[vaxmedcd2[i]][d] << "\t";
		}
		clustcd2_vaxmed_vx_file << std::endl;
		clustcd2_vaxmed_file << std::endl;
	}
	std::string clustcd2_vaxhigh_filename;
	clustcd2_vaxhigh_filename = s + "_clustcd2_vaxhigh.txt";
	std::ofstream clustcd2_vaxhigh_file;
	clustcd2_vaxhigh_file.open(clustcd2_vaxhigh_filename);
	std::string clustcd2_vaxhigh_vx_filename;
	clustcd2_vaxhigh_vx_filename = s + "_clustcd2_vaxhigh_vx.txt";
	std::ofstream clustcd2_vaxhigh_vx_file;
	clustcd2_vaxhigh_vx_file.open(clustcd2_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd2.size(); i++) {
			clustcd2_vaxhigh_vx_file << vaccinations[vaxhighcd2[i]][d] << "\t";
			clustcd2_vaxhigh_file << coviddeaths[vaxhighcd2[i]][d] << "\t";
		}
		clustcd2_vaxhigh_vx_file << std::endl;
		clustcd2_vaxhigh_file << std::endl;
	}

	std::string clustcd3_vaxlow_filename;
	clustcd3_vaxlow_filename = s + "_clustcd3_vaxlow.txt";
	std::ofstream clustcd3_vaxlow_file;
	clustcd3_vaxlow_file.open(clustcd3_vaxlow_filename);
	std::string clustcd3_vaxlow_vx_filename;
	clustcd3_vaxlow_vx_filename = s + "_clustcd3_vaxlow_vx.txt";
	std::ofstream clustcd3_vaxlow_vx_file;
	clustcd3_vaxlow_vx_file.open(clustcd3_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd3.size(); i++) {
			clustcd3_vaxlow_vx_file << vaccinations[vaxlowcd3[i]][d] << "\t";
			clustcd3_vaxlow_file << coviddeaths[vaxlowcd3[i]][d] << "\t";
		}
		clustcd3_vaxlow_vx_file << std::endl;
		clustcd3_vaxlow_file << std::endl;
	}
	std::string clustcd3_vaxmed_filename;
	clustcd3_vaxmed_filename = s + "_clustcd3_vaxmed.txt";
	std::ofstream clustcd3_vaxmed_file;
	clustcd3_vaxmed_file.open(clustcd3_vaxmed_filename);
	std::string clustcd3_vaxmed_vx_filename;
	clustcd3_vaxmed_vx_filename = s + "_clustcd3_vaxmed_vx.txt";
	std::ofstream clustcd3_vaxmed_vx_file;
	clustcd3_vaxmed_vx_file.open(clustcd3_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd3.size(); i++) {
			clustcd3_vaxmed_vx_file << vaccinations[vaxmedcd3[i]][d] << "\t";
			clustcd3_vaxmed_file << coviddeaths[vaxmedcd3[i]][d] << "\t";
		}
		clustcd3_vaxmed_vx_file << std::endl;
		clustcd3_vaxmed_file << std::endl;
	}
	std::string clustcd3_vaxhigh_filename;
	clustcd3_vaxhigh_filename = s + "_clustcd3_vaxhigh.txt";
	std::ofstream clustcd3_vaxhigh_file;
	clustcd3_vaxhigh_file.open(clustcd3_vaxhigh_filename);
	std::string clustcd3_vaxhigh_vx_filename;
	clustcd3_vaxhigh_vx_filename = s + "_clustcd3_vaxhigh_vx.txt";
	std::ofstream clustcd3_vaxhigh_vx_file;
	clustcd3_vaxhigh_vx_file.open(clustcd3_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd3.size(); i++) {
			clustcd3_vaxhigh_vx_file << vaccinations[vaxhighcd3[i]][d] << "\t";
			clustcd3_vaxhigh_file << coviddeaths[vaxhighcd3[i]][d] << "\t";
		}
		clustcd3_vaxhigh_vx_file << std::endl;
		clustcd3_vaxhigh_file << std::endl;
	}

	std::string clustcd4_vaxlow_filename;
	clustcd4_vaxlow_filename = s + "_clustcd4_vaxlow.txt";
	std::ofstream clustcd4_vaxlow_file;
	clustcd4_vaxlow_file.open(clustcd4_vaxlow_filename);
	std::string clustcd4_vaxlow_vx_filename;
	clustcd4_vaxlow_vx_filename = s + "_clustcd4_vaxlow_vx.txt";
	std::ofstream clustcd4_vaxlow_vx_file;
	clustcd4_vaxlow_vx_file.open(clustcd4_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd4.size(); i++) {
			clustcd4_vaxlow_vx_file << vaccinations[vaxlowcd4[i]][d] << "\t";
			clustcd4_vaxlow_file << coviddeaths[vaxlowcd4[i]][d] << "\t";
		}
		clustcd4_vaxlow_vx_file << std::endl;
		clustcd4_vaxlow_file << std::endl;
	}
	std::string clustcd4_vaxmed_filename;
	clustcd4_vaxmed_filename = s + "_clustcd4_vaxmed.txt";
	std::ofstream clustcd4_vaxmed_file;
	clustcd4_vaxmed_file.open(clustcd4_vaxmed_filename);
	std::string clustcd4_vaxmed_vx_filename;
	clustcd4_vaxmed_vx_filename = s + "_clustcd4_vaxmed_vx.txt";
	std::ofstream clustcd4_vaxmed_vx_file;
	clustcd4_vaxmed_vx_file.open(clustcd4_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd4.size(); i++) {
			clustcd4_vaxmed_vx_file << vaccinations[vaxmedcd4[i]][d] << "\t";
			clustcd4_vaxmed_file << coviddeaths[vaxmedcd4[i]][d] << "\t";
		}
		clustcd4_vaxmed_vx_file << std::endl;
		clustcd4_vaxmed_file << std::endl;
	}
	std::string clustcd4_vaxhigh_filename;
	clustcd4_vaxhigh_filename = s + "_clustcd4_vaxhigh.txt";
	std::ofstream clustcd4_vaxhigh_file;
	clustcd4_vaxhigh_file.open(clustcd4_vaxhigh_filename);
	std::string clustcd4_vaxhigh_vx_filename;
	clustcd4_vaxhigh_vx_filename = s + "_clustcd4_vaxhigh_vx.txt";
	std::ofstream clustcd4_vaxhigh_vx_file;
	clustcd4_vaxhigh_vx_file.open(clustcd4_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd4.size(); i++) {
			clustcd4_vaxhigh_vx_file << vaccinations[vaxhighcd4[i]][d] << "\t";
			clustcd4_vaxhigh_file << coviddeaths[vaxhighcd4[i]][d] << "\t";
		}
		clustcd4_vaxhigh_vx_file << std::endl;
		clustcd4_vaxhigh_file << std::endl;
	}

	std::string clustcd5_vaxlow_filename;
	clustcd5_vaxlow_filename = s + "_clustcd5_vaxlow.txt";
	std::ofstream clustcd5_vaxlow_file;
	clustcd5_vaxlow_file.open(clustcd5_vaxlow_filename);
	std::string clustcd5_vaxlow_vx_filename;
	clustcd5_vaxlow_vx_filename = s + "_clustcd5_vaxlow_vx.txt";
	std::ofstream clustcd5_vaxlow_vx_file;
	clustcd5_vaxlow_vx_file.open(clustcd5_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd5.size(); i++) {
			clustcd5_vaxlow_vx_file << vaccinations[vaxlowcd5[i]][d] << "\t";
			clustcd5_vaxlow_file << coviddeaths[vaxlowcd5[i]][d] << "\t";
		}
		clustcd5_vaxlow_vx_file << std::endl;
		clustcd5_vaxlow_file << std::endl;
	}
	std::string clustcd5_vaxmed_filename;
	clustcd5_vaxmed_filename = s + "_clustcd5_vaxmed.txt";
	std::ofstream clustcd5_vaxmed_file;
	clustcd5_vaxmed_file.open(clustcd5_vaxmed_filename);
	std::string clustcd5_vaxmed_vx_filename;
	clustcd5_vaxmed_vx_filename = s + "_clustcd5_vaxmed_vx.txt";
	std::ofstream clustcd5_vaxmed_vx_file;
	clustcd5_vaxmed_vx_file.open(clustcd5_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd5.size(); i++) {
			clustcd5_vaxmed_vx_file << vaccinations[vaxmedcd5[i]][d] << "\t";
			clustcd5_vaxmed_file << coviddeaths[vaxmedcd5[i]][d] << "\t";
		}
		clustcd5_vaxmed_vx_file << std::endl;
		clustcd5_vaxmed_file << std::endl;
	}
	std::string clustcd5_vaxhigh_filename;
	clustcd5_vaxhigh_filename = s + "_clustcd5_vaxhigh.txt";
	std::ofstream clustcd5_vaxhigh_file;
	clustcd5_vaxhigh_file.open(clustcd5_vaxhigh_filename);
	std::string clustcd5_vaxhigh_vx_filename;
	clustcd5_vaxhigh_vx_filename = s + "_clustcd5_vaxhigh_vx.txt";
	std::ofstream clustcd5_vaxhigh_vx_file;
	clustcd5_vaxhigh_vx_file.open(clustcd5_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd5.size(); i++) {
			clustcd5_vaxhigh_vx_file << vaccinations[vaxhighcd5[i]][d] << "\t";
			clustcd5_vaxhigh_file << coviddeaths[vaxhighcd5[i]][d] << "\t";
		}
		clustcd5_vaxhigh_vx_file << std::endl;
		clustcd5_vaxhigh_file << std::endl;
	}

	std::string clustcd6_vaxlow_filename;
	clustcd6_vaxlow_filename = s + "_clustcd6_vaxlow.txt";
	std::ofstream clustcd6_vaxlow_file;
	clustcd6_vaxlow_file.open(clustcd6_vaxlow_filename);
	std::string clustcd6_vaxlow_vx_filename;
	clustcd6_vaxlow_vx_filename = s + "_clustcd6_vaxlow_vx.txt";
	std::ofstream clustcd6_vaxlow_vx_file;
	clustcd6_vaxlow_vx_file.open(clustcd6_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd6.size(); i++) {
			clustcd6_vaxlow_vx_file << vaccinations[vaxlowcd6[i]][d] << "\t";
			clustcd6_vaxlow_file << coviddeaths[vaxlowcd6[i]][d] << "\t";
		}
		clustcd6_vaxlow_vx_file << std::endl;
		clustcd6_vaxlow_file << std::endl;
	}
	std::string clustcd6_vaxmed_filename;
	clustcd6_vaxmed_filename = s + "_clustcd6_vaxmed.txt";
	std::ofstream clustcd6_vaxmed_file;
	clustcd6_vaxmed_file.open(clustcd6_vaxmed_filename);
	std::string clustcd6_vaxmed_vx_filename;
	clustcd6_vaxmed_vx_filename = s + "_clustcd6_vaxmed_vx.txt";
	std::ofstream clustcd6_vaxmed_vx_file;
	clustcd6_vaxmed_vx_file.open(clustcd6_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd6.size(); i++) {
			clustcd6_vaxmed_vx_file << vaccinations[vaxmedcd6[i]][d] << "\t";
			clustcd6_vaxmed_file << coviddeaths[vaxmedcd6[i]][d] << "\t";
		}
		clustcd6_vaxmed_vx_file << std::endl;
		clustcd6_vaxmed_file << std::endl;
	}
	std::string clustcd6_vaxhigh_filename;
	clustcd6_vaxhigh_filename = s + "_clustcd6_vaxhigh.txt";
	std::ofstream clustcd6_vaxhigh_file;
	clustcd6_vaxhigh_file.open(clustcd6_vaxhigh_filename);
	std::string clustcd6_vaxhigh_vx_filename;
	clustcd6_vaxhigh_vx_filename = s + "_clustcd6_vaxhigh_vx.txt";
	std::ofstream clustcd6_vaxhigh_vx_file;
	clustcd6_vaxhigh_vx_file.open(clustcd6_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd6.size(); i++) {
			clustcd6_vaxhigh_vx_file << vaccinations[vaxhighcd6[i]][d] << "\t";
			clustcd6_vaxhigh_file << coviddeaths[vaxhighcd6[i]][d] << "\t";
		}
		clustcd6_vaxhigh_vx_file << std::endl;
		clustcd6_vaxhigh_file << std::endl;
	}

	std::string clustcd7_vaxlow_filename;
	clustcd7_vaxlow_filename = s + "_clustcd7_vaxlow.txt";
	std::ofstream clustcd7_vaxlow_file;
	clustcd7_vaxlow_file.open(clustcd7_vaxlow_filename);
	std::string clustcd7_vaxlow_vx_filename;
	clustcd7_vaxlow_vx_filename = s + "_clustcd7_vaxlow_vx.txt";
	std::ofstream clustcd7_vaxlow_vx_file;
	clustcd7_vaxlow_vx_file.open(clustcd7_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd7.size(); i++) {
			clustcd7_vaxlow_vx_file << vaccinations[vaxlowcd7[i]][d] << "\t";
			clustcd7_vaxlow_file << coviddeaths[vaxlowcd7[i]][d] << "\t";
		}
		clustcd7_vaxlow_vx_file << std::endl;
		clustcd7_vaxlow_file << std::endl;
	}
	std::string clustcd7_vaxmed_filename;
	clustcd7_vaxmed_filename = s + "_clustcd7_vaxmed.txt";
	std::ofstream clustcd7_vaxmed_file;
	clustcd7_vaxmed_file.open(clustcd7_vaxmed_filename);
	std::string clustcd7_vaxmed_vx_filename;
	clustcd7_vaxmed_vx_filename = s + "_clustcd7_vaxmed_vx.txt";
	std::ofstream clustcd7_vaxmed_vx_file;
	clustcd7_vaxmed_vx_file.open(clustcd7_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd7.size(); i++) {
			clustcd7_vaxmed_vx_file << vaccinations[vaxmedcd7[i]][d] << "\t";
			clustcd7_vaxmed_file << coviddeaths[vaxmedcd7[i]][d] << "\t";
		}
		clustcd7_vaxmed_vx_file << std::endl;
		clustcd7_vaxmed_file << std::endl;
	}
	std::string clustcd7_vaxhigh_filename;
	clustcd7_vaxhigh_filename = s + "_clustcd7_vaxhigh.txt";
	std::ofstream clustcd7_vaxhigh_file;
	clustcd7_vaxhigh_file.open(clustcd7_vaxhigh_filename);
	std::string clustcd7_vaxhigh_vx_filename;
	clustcd7_vaxhigh_vx_filename = s + "_clustcd7_vaxhigh_vx.txt";
	std::ofstream clustcd7_vaxhigh_vx_file;
	clustcd7_vaxhigh_vx_file.open(clustcd7_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd7.size(); i++) {
			clustcd7_vaxhigh_vx_file << vaccinations[vaxhighcd7[i]][d] << "\t";
			clustcd7_vaxhigh_file << coviddeaths[vaxhighcd7[i]][d] << "\t";
		}
		clustcd7_vaxhigh_vx_file << std::endl;
		clustcd7_vaxhigh_file << std::endl;
	}

	std::string clustcd8_vaxlow_filename;
	clustcd8_vaxlow_filename = s + "_clustcd8_vaxlow.txt";
	std::ofstream clustcd8_vaxlow_file;
	clustcd8_vaxlow_file.open(clustcd8_vaxlow_filename);
	std::string clustcd8_vaxlow_vx_filename;
	clustcd8_vaxlow_vx_filename = s + "_clustcd8_vaxlow_vx.txt";
	std::ofstream clustcd8_vaxlow_vx_file;
	clustcd8_vaxlow_vx_file.open(clustcd8_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd8.size(); i++) {
			clustcd8_vaxlow_vx_file << vaccinations[vaxlowcd8[i]][d] << "\t";
			clustcd8_vaxlow_file << coviddeaths[vaxlowcd8[i]][d] << "\t";
		}
		clustcd8_vaxlow_vx_file << std::endl;
		clustcd8_vaxlow_file << std::endl;
	}
	std::string clustcd8_vaxmed_filename;
	clustcd8_vaxmed_filename = s + "_clustcd8_vaxmed.txt";
	std::ofstream clustcd8_vaxmed_file;
	clustcd8_vaxmed_file.open(clustcd8_vaxmed_filename);
	std::string clustcd8_vaxmed_vx_filename;
	clustcd8_vaxmed_vx_filename = s + "_clustcd8_vaxmed_vx.txt";
	std::ofstream clustcd8_vaxmed_vx_file;
	clustcd8_vaxmed_vx_file.open(clustcd8_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd8.size(); i++) {
			clustcd8_vaxmed_vx_file << vaccinations[vaxmedcd8[i]][d] << "\t";
			clustcd8_vaxmed_file << coviddeaths[vaxmedcd8[i]][d] << "\t";
		}
		clustcd8_vaxmed_vx_file << std::endl;
		clustcd8_vaxmed_file << std::endl;
	}
	std::string clustcd8_vaxhigh_filename;
	clustcd8_vaxhigh_filename = s + "_clustcd8_vaxhigh.txt";
	std::ofstream clustcd8_vaxhigh_file;
	clustcd8_vaxhigh_file.open(clustcd8_vaxhigh_filename);
	std::string clustcd8_vaxhigh_vx_filename;
	clustcd8_vaxhigh_vx_filename = s + "_clustcd8_vaxhigh_vx.txt";
	std::ofstream clustcd8_vaxhigh_vx_file;
	clustcd8_vaxhigh_vx_file.open(clustcd8_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd8.size(); i++) {
			clustcd8_vaxhigh_vx_file << vaccinations[vaxhighcd8[i]][d] << "\t";
			clustcd8_vaxhigh_file << coviddeaths[vaxhighcd8[i]][d] << "\t";
		}
		clustcd8_vaxhigh_vx_file << std::endl;
		clustcd8_vaxhigh_file << std::endl;
	}

	std::string clustcd9_vaxlow_filename;
	clustcd9_vaxlow_filename = s + "_clustcd9_vaxlow.txt";
	std::ofstream clustcd9_vaxlow_file;
	clustcd9_vaxlow_file.open(clustcd9_vaxlow_filename);
	std::string clustcd9_vaxlow_vx_filename;
	clustcd9_vaxlow_vx_filename = s + "_clustcd9_vaxlow_vx.txt";
	std::ofstream clustcd9_vaxlow_vx_file;
	clustcd9_vaxlow_vx_file.open(clustcd9_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd9.size(); i++) {
			clustcd9_vaxlow_vx_file << vaccinations[vaxlowcd9[i]][d] << "\t";
			clustcd9_vaxlow_file << coviddeaths[vaxlowcd9[i]][d] << "\t";
		}
		clustcd9_vaxlow_vx_file << std::endl;
		clustcd9_vaxlow_file << std::endl;
	}
	std::string clustcd9_vaxmed_filename;
	clustcd9_vaxmed_filename = s + "_clustcd9_vaxmed.txt";
	std::ofstream clustcd9_vaxmed_file;
	clustcd9_vaxmed_file.open(clustcd9_vaxmed_filename);
	std::string clustcd9_vaxmed_vx_filename;
	clustcd9_vaxmed_vx_filename = s + "_clustcd9_vaxmed_vx.txt";
	std::ofstream clustcd9_vaxmed_vx_file;
	clustcd9_vaxmed_vx_file.open(clustcd9_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd9.size(); i++) {
			clustcd9_vaxmed_vx_file << vaccinations[vaxmedcd9[i]][d] << "\t";
			clustcd9_vaxmed_file << coviddeaths[vaxmedcd9[i]][d] << "\t";
		}
		clustcd9_vaxmed_vx_file << std::endl;
		clustcd9_vaxmed_file << std::endl;
	}
	std::string clustcd9_vaxhigh_filename;
	clustcd9_vaxhigh_filename = s + "_clustcd9_vaxhigh.txt";
	std::ofstream clustcd9_vaxhigh_file;
	clustcd9_vaxhigh_file.open(clustcd9_vaxhigh_filename);
	std::string clustcd9_vaxhigh_vx_filename;
	clustcd9_vaxhigh_vx_filename = s + "_clustcd9_vaxhigh_vx.txt";
	std::ofstream clustcd9_vaxhigh_vx_file;
	clustcd9_vaxhigh_vx_file.open(clustcd9_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd9.size(); i++) {
			clustcd9_vaxhigh_vx_file << vaccinations[vaxhighcd9[i]][d] << "\t";
			clustcd9_vaxhigh_file << coviddeaths[vaxhighcd9[i]][d] << "\t";
		}
		clustcd9_vaxhigh_vx_file << std::endl;
		clustcd9_vaxhigh_file << std::endl;
	}

	std::string clustcd10_vaxlow_filename;
	clustcd10_vaxlow_filename = s + "_clustcd10_vaxlow.txt";
	std::ofstream clustcd10_vaxlow_file;
	clustcd10_vaxlow_file.open(clustcd10_vaxlow_filename);
	std::string clustcd10_vaxlow_vx_filename;
	clustcd10_vaxlow_vx_filename = s + "_clustcd10_vaxlow_vx.txt";
	std::ofstream clustcd10_vaxlow_vx_file;
	clustcd10_vaxlow_vx_file.open(clustcd10_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd10.size(); i++) {
			clustcd10_vaxlow_vx_file << vaccinations[vaxlowcd10[i]][d] << "\t";
			clustcd10_vaxlow_file << coviddeaths[vaxlowcd10[i]][d] << "\t";
		}
		clustcd10_vaxlow_vx_file << std::endl;
		clustcd10_vaxlow_file << std::endl;
	}
	std::string clustcd10_vaxmed_filename;
	clustcd10_vaxmed_filename = s + "_clustcd10_vaxmed.txt";
	std::ofstream clustcd10_vaxmed_file;
	clustcd10_vaxmed_file.open(clustcd10_vaxmed_filename);
	std::string clustcd10_vaxmed_vx_filename;
	clustcd10_vaxmed_vx_filename = s + "_clustcd10_vaxmed_vx.txt";
	std::ofstream clustcd10_vaxmed_vx_file;
	clustcd10_vaxmed_vx_file.open(clustcd10_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd10.size(); i++) {
			clustcd10_vaxmed_vx_file << vaccinations[vaxmedcd10[i]][d] << "\t";
			clustcd10_vaxmed_file << coviddeaths[vaxmedcd10[i]][d] << "\t";
		}
		clustcd10_vaxmed_vx_file << std::endl;
		clustcd10_vaxmed_file << std::endl;
	}
	std::string clustcd10_vaxhigh_filename;
	clustcd10_vaxhigh_filename = s + "_clustcd10_vaxhigh.txt";
	std::ofstream clustcd10_vaxhigh_file;
	clustcd10_vaxhigh_file.open(clustcd10_vaxhigh_filename);
	std::string clustcd10_vaxhigh_vx_filename;
	clustcd10_vaxhigh_vx_filename = s + "_clustcd10_vaxhigh_vx.txt";
	std::ofstream clustcd10_vaxhigh_vx_file;
	clustcd10_vaxhigh_vx_file.open(clustcd10_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd10.size(); i++) {
			clustcd10_vaxhigh_vx_file << vaccinations[vaxhighcd10[i]][d] << "\t";
			clustcd10_vaxhigh_file << coviddeaths[vaxhighcd10[i]][d] << "\t";
		}
		clustcd10_vaxhigh_vx_file << std::endl;
		clustcd10_vaxhigh_file << std::endl;
	}

	std::string clustcd11_vaxlow_filename;
	clustcd11_vaxlow_filename = s + "_clustcd11_vaxlow.txt";
	std::ofstream clustcd11_vaxlow_file;
	clustcd11_vaxlow_file.open(clustcd11_vaxlow_filename);
	std::string clustcd11_vaxlow_vx_filename;
	clustcd11_vaxlow_vx_filename = s + "_clustcd11_vaxlow_vx.txt";
	std::ofstream clustcd11_vaxlow_vx_file;
	clustcd11_vaxlow_vx_file.open(clustcd11_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd11.size(); i++) {
			clustcd11_vaxlow_vx_file << vaccinations[vaxlowcd11[i]][d] << "\t";
			clustcd11_vaxlow_file << coviddeaths[vaxlowcd11[i]][d] << "\t";
		}
		clustcd11_vaxlow_vx_file << std::endl;
		clustcd11_vaxlow_file << std::endl;
	}
	std::string clustcd11_vaxmed_filename;
	clustcd11_vaxmed_filename = s + "_clustcd11_vaxmed.txt";
	std::ofstream clustcd11_vaxmed_file;
	clustcd11_vaxmed_file.open(clustcd11_vaxmed_filename);
	std::string clustcd11_vaxmed_vx_filename;
	clustcd11_vaxmed_vx_filename = s + "_clustcd11_vaxmed_vx.txt";
	std::ofstream clustcd11_vaxmed_vx_file;
	clustcd11_vaxmed_vx_file.open(clustcd11_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd11.size(); i++) {
			clustcd11_vaxmed_vx_file << vaccinations[vaxmedcd11[i]][d] << "\t";
			clustcd11_vaxmed_file << coviddeaths[vaxmedcd11[i]][d] << "\t";
		}
		clustcd11_vaxmed_vx_file << std::endl;
		clustcd11_vaxmed_file << std::endl;
	}
	std::string clustcd11_vaxhigh_filename;
	clustcd11_vaxhigh_filename = s + "_clustcd11_vaxhigh.txt";
	std::ofstream clustcd11_vaxhigh_file;
	clustcd11_vaxhigh_file.open(clustcd11_vaxhigh_filename);
	std::string clustcd11_vaxhigh_vx_filename;
	clustcd11_vaxhigh_vx_filename = s + "_clustcd11_vaxhigh_vx.txt";
	std::ofstream clustcd11_vaxhigh_vx_file;
	clustcd11_vaxhigh_vx_file.open(clustcd11_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd11.size(); i++) {
			clustcd11_vaxhigh_vx_file << vaccinations[vaxhighcd11[i]][d] << "\t";
			clustcd11_vaxhigh_file << coviddeaths[vaxhighcd11[i]][d] << "\t";
		}
		clustcd11_vaxhigh_vx_file << std::endl;
		clustcd11_vaxhigh_file << std::endl;
	}

	std::string clustcd12_vaxlow_filename;
	clustcd12_vaxlow_filename = s + "_clustcd12_vaxlow.txt";
	std::ofstream clustcd12_vaxlow_file;
	clustcd12_vaxlow_file.open(clustcd12_vaxlow_filename);
	std::string clustcd12_vaxlow_vx_filename;
	clustcd12_vaxlow_vx_filename = s + "_clustcd12_vaxlow_vx.txt";
	std::ofstream clustcd12_vaxlow_vx_file;
	clustcd12_vaxlow_vx_file.open(clustcd12_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd12.size(); i++) {
			clustcd12_vaxlow_vx_file << vaccinations[vaxlowcd12[i]][d] << "\t";
			clustcd12_vaxlow_file << coviddeaths[vaxlowcd12[i]][d] << "\t";
		}
		clustcd12_vaxlow_vx_file << std::endl;
		clustcd12_vaxlow_file << std::endl;
	}
	std::string clustcd12_vaxmed_filename;
	clustcd12_vaxmed_filename = s + "_clustcd12_vaxmed.txt";
	std::ofstream clustcd12_vaxmed_file;
	clustcd12_vaxmed_file.open(clustcd12_vaxmed_filename);
	std::string clustcd12_vaxmed_vx_filename;
	clustcd12_vaxmed_vx_filename = s + "_clustcd12_vaxmed_vx.txt";
	std::ofstream clustcd12_vaxmed_vx_file;
	clustcd12_vaxmed_vx_file.open(clustcd12_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd12.size(); i++) {
			clustcd12_vaxmed_vx_file << vaccinations[vaxmedcd12[i]][d] << "\t";
			clustcd12_vaxmed_file << coviddeaths[vaxmedcd12[i]][d] << "\t";
		}
		clustcd12_vaxmed_vx_file << std::endl;
		clustcd12_vaxmed_file << std::endl;
	}
	std::string clustcd12_vaxhigh_filename;
	clustcd12_vaxhigh_filename = s + "_clustcd12_vaxhigh.txt";
	std::ofstream clustcd12_vaxhigh_file;
	clustcd12_vaxhigh_file.open(clustcd12_vaxhigh_filename);
	std::string clustcd12_vaxhigh_vx_filename;
	clustcd12_vaxhigh_vx_filename = s + "_clustcd12_vaxhigh_vx.txt";
	std::ofstream clustcd12_vaxhigh_vx_file;
	clustcd12_vaxhigh_vx_file.open(clustcd12_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd12.size(); i++) {
			clustcd12_vaxhigh_vx_file << vaccinations[vaxhighcd12[i]][d] << "\t";
			clustcd12_vaxhigh_file << coviddeaths[vaxhighcd12[i]][d] << "\t";
		}
		clustcd12_vaxhigh_vx_file << std::endl;
		clustcd12_vaxhigh_file << std::endl;
	}

	std::string clustcd13_vaxlow_filename;
	clustcd13_vaxlow_filename = s + "_clustcd13_vaxlow.txt";
	std::ofstream clustcd13_vaxlow_file;
	clustcd13_vaxlow_file.open(clustcd13_vaxlow_filename);
	std::string clustcd13_vaxlow_vx_filename;
	clustcd13_vaxlow_vx_filename = s + "_clustcd13_vaxlow_vx.txt";
	std::ofstream clustcd13_vaxlow_vx_file;
	clustcd13_vaxlow_vx_file.open(clustcd13_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd13.size(); i++) {
			clustcd13_vaxlow_vx_file << vaccinations[vaxlowcd13[i]][d] << "\t";
			clustcd13_vaxlow_file << coviddeaths[vaxlowcd13[i]][d] << "\t";
		}
		clustcd13_vaxlow_vx_file << std::endl;
		clustcd13_vaxlow_file << std::endl;
	}
	std::string clustcd13_vaxmed_filename;
	clustcd13_vaxmed_filename = s + "_clustcd13_vaxmed.txt";
	std::ofstream clustcd13_vaxmed_file;
	clustcd13_vaxmed_file.open(clustcd13_vaxmed_filename);
	std::string clustcd13_vaxmed_vx_filename;
	clustcd13_vaxmed_vx_filename = s + "_clustcd13_vaxmed_vx.txt";
	std::ofstream clustcd13_vaxmed_vx_file;
	clustcd13_vaxmed_vx_file.open(clustcd13_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd13.size(); i++) {
			clustcd13_vaxmed_vx_file << vaccinations[vaxmedcd13[i]][d] << "\t";
			clustcd13_vaxmed_file << coviddeaths[vaxmedcd13[i]][d] << "\t";
		}
		clustcd13_vaxmed_vx_file << std::endl;
		clustcd13_vaxmed_file << std::endl;
	}
	std::string clustcd13_vaxhigh_filename;
	clustcd13_vaxhigh_filename = s + "_clustcd13_vaxhigh.txt";
	std::ofstream clustcd13_vaxhigh_file;
	clustcd13_vaxhigh_file.open(clustcd13_vaxhigh_filename);
	std::string clustcd13_vaxhigh_vx_filename;
	clustcd13_vaxhigh_vx_filename = s + "_clustcd13_vaxhigh_vx.txt";
	std::ofstream clustcd13_vaxhigh_vx_file;
	clustcd13_vaxhigh_vx_file.open(clustcd13_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd13.size(); i++) {
			clustcd13_vaxhigh_vx_file << vaccinations[vaxhighcd13[i]][d] << "\t";
			clustcd13_vaxhigh_file << coviddeaths[vaxhighcd13[i]][d] << "\t";
		}
		clustcd13_vaxhigh_vx_file << std::endl;
		clustcd13_vaxhigh_file << std::endl;
	}

	std::string clustcd14_vaxlow_filename;
	clustcd14_vaxlow_filename = s + "_clustcd14_vaxlow.txt";
	std::ofstream clustcd14_vaxlow_file;
	clustcd14_vaxlow_file.open(clustcd14_vaxlow_filename);
	std::string clustcd14_vaxlow_vx_filename;
	clustcd14_vaxlow_vx_filename = s + "_clustcd14_vaxlow_vx.txt";
	std::ofstream clustcd14_vaxlow_vx_file;
	clustcd14_vaxlow_vx_file.open(clustcd14_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd14.size(); i++) {
			clustcd14_vaxlow_vx_file << vaccinations[vaxlowcd14[i]][d] << "\t";
			clustcd14_vaxlow_file << coviddeaths[vaxlowcd14[i]][d] << "\t";
		}
		clustcd14_vaxlow_vx_file << std::endl;
		clustcd14_vaxlow_file << std::endl;
	}
	std::string clustcd14_vaxmed_filename;
	clustcd14_vaxmed_filename = s + "_clustcd14_vaxmed.txt";
	std::ofstream clustcd14_vaxmed_file;
	clustcd14_vaxmed_file.open(clustcd14_vaxmed_filename);
	std::string clustcd14_vaxmed_vx_filename;
	clustcd14_vaxmed_vx_filename = s + "_clustcd14_vaxmed_vx.txt";
	std::ofstream clustcd14_vaxmed_vx_file;
	clustcd14_vaxmed_vx_file.open(clustcd14_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd14.size(); i++) {
			clustcd14_vaxmed_vx_file << vaccinations[vaxmedcd14[i]][d] << "\t";
			clustcd14_vaxmed_file << coviddeaths[vaxmedcd14[i]][d] << "\t";
		}
		clustcd14_vaxmed_vx_file << std::endl;
		clustcd14_vaxmed_file << std::endl;
	}
	std::string clustcd14_vaxhigh_filename;
	clustcd14_vaxhigh_filename = s + "_clustcd14_vaxhigh.txt";
	std::ofstream clustcd14_vaxhigh_file;
	clustcd14_vaxhigh_file.open(clustcd14_vaxhigh_filename);
	std::string clustcd14_vaxhigh_vx_filename;
	clustcd14_vaxhigh_vx_filename = s + "_clustcd14_vaxhigh_vx.txt";
	std::ofstream clustcd14_vaxhigh_vx_file;
	clustcd14_vaxhigh_vx_file.open(clustcd14_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd14.size(); i++) {
			clustcd14_vaxhigh_vx_file << vaccinations[vaxhighcd14[i]][d] << "\t";
			clustcd14_vaxhigh_file << coviddeaths[vaxhighcd14[i]][d] << "\t";
		}
		clustcd14_vaxhigh_vx_file << std::endl;
		clustcd14_vaxhigh_file << std::endl;
	}

	std::string clustcd15_vaxlow_filename;
	clustcd15_vaxlow_filename = s + "_clustcd15_vaxlow.txt";
	std::ofstream clustcd15_vaxlow_file;
	clustcd15_vaxlow_file.open(clustcd15_vaxlow_filename);
	std::string clustcd15_vaxlow_vx_filename;
	clustcd15_vaxlow_vx_filename = s + "_clustcd15_vaxlow_vx.txt";
	std::ofstream clustcd15_vaxlow_vx_file;
	clustcd15_vaxlow_vx_file.open(clustcd15_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd15.size(); i++) {
			clustcd15_vaxlow_vx_file << vaccinations[vaxlowcd15[i]][d] << "\t";
			clustcd15_vaxlow_file << coviddeaths[vaxlowcd15[i]][d] << "\t";
		}
		clustcd15_vaxlow_vx_file << std::endl;
		clustcd15_vaxlow_file << std::endl;
	}
	std::string clustcd15_vaxmed_filename;
	clustcd15_vaxmed_filename = s + "_clustcd15_vaxmed.txt";
	std::ofstream clustcd15_vaxmed_file;
	clustcd15_vaxmed_file.open(clustcd15_vaxmed_filename);
	std::string clustcd15_vaxmed_vx_filename;
	clustcd15_vaxmed_vx_filename = s + "_clustcd15_vaxmed_vx.txt";
	std::ofstream clustcd15_vaxmed_vx_file;
	clustcd15_vaxmed_vx_file.open(clustcd15_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd15.size(); i++) {
			clustcd15_vaxmed_vx_file << vaccinations[vaxmedcd15[i]][d] << "\t";
			clustcd15_vaxmed_file << coviddeaths[vaxmedcd15[i]][d] << "\t";
		}
		clustcd15_vaxmed_vx_file << std::endl;
		clustcd15_vaxmed_file << std::endl;
	}
	std::string clustcd15_vaxhigh_filename;
	clustcd15_vaxhigh_filename = s + "_clustcd15_vaxhigh.txt";
	std::ofstream clustcd15_vaxhigh_file;
	clustcd15_vaxhigh_file.open(clustcd15_vaxhigh_filename);
	std::string clustcd15_vaxhigh_vx_filename;
	clustcd15_vaxhigh_vx_filename = s + "_clustcd15_vaxhigh_vx.txt";
	std::ofstream clustcd15_vaxhigh_vx_file;
	clustcd15_vaxhigh_vx_file.open(clustcd15_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd15.size(); i++) {
			clustcd15_vaxhigh_vx_file << vaccinations[vaxhighcd15[i]][d] << "\t";
			clustcd15_vaxhigh_file << coviddeaths[vaxhighcd15[i]][d] << "\t";
		}
		clustcd15_vaxhigh_vx_file << std::endl;
		clustcd15_vaxhigh_file << std::endl;
	}

	std::string clustcd16_vaxlow_filename;
	clustcd16_vaxlow_filename = s + "_clustcd16_vaxlow.txt";
	std::ofstream clustcd16_vaxlow_file;
	clustcd16_vaxlow_file.open(clustcd16_vaxlow_filename);
	std::string clustcd16_vaxlow_vx_filename;
	clustcd16_vaxlow_vx_filename = s + "_clustcd16_vaxlow_vx.txt";
	std::ofstream clustcd16_vaxlow_vx_file;
	clustcd16_vaxlow_vx_file.open(clustcd16_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd16.size(); i++) {
			clustcd16_vaxlow_vx_file << vaccinations[vaxlowcd16[i]][d] << "\t";
			clustcd16_vaxlow_file << coviddeaths[vaxlowcd16[i]][d] << "\t";
		}
		clustcd16_vaxlow_vx_file << std::endl;
		clustcd16_vaxlow_file << std::endl;
	}
	std::string clustcd16_vaxmed_filename;
	clustcd16_vaxmed_filename = s + "_clustcd16_vaxmed.txt";
	std::ofstream clustcd16_vaxmed_file;
	clustcd16_vaxmed_file.open(clustcd16_vaxmed_filename);
	std::string clustcd16_vaxmed_vx_filename;
	clustcd16_vaxmed_vx_filename = s + "_clustcd16_vaxmed_vx.txt";
	std::ofstream clustcd16_vaxmed_vx_file;
	clustcd16_vaxmed_vx_file.open(clustcd16_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd16.size(); i++) {
			clustcd16_vaxmed_vx_file << vaccinations[vaxmedcd16[i]][d] << "\t";
			clustcd16_vaxmed_file << coviddeaths[vaxmedcd16[i]][d] << "\t";
		}
		clustcd16_vaxmed_vx_file << std::endl;
		clustcd16_vaxmed_file << std::endl;
	}
	std::string clustcd16_vaxhigh_filename;
	clustcd16_vaxhigh_filename = s + "_clustcd16_vaxhigh.txt";
	std::ofstream clustcd16_vaxhigh_file;
	clustcd16_vaxhigh_file.open(clustcd16_vaxhigh_filename);
	std::string clustcd16_vaxhigh_vx_filename;
	clustcd16_vaxhigh_vx_filename = s + "_clustcd16_vaxhigh_vx.txt";
	std::ofstream clustcd16_vaxhigh_vx_file;
	clustcd16_vaxhigh_vx_file.open(clustcd16_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd16.size(); i++) {
			clustcd16_vaxhigh_vx_file << vaccinations[vaxhighcd16[i]][d] << "\t";
			clustcd16_vaxhigh_file << coviddeaths[vaxhighcd16[i]][d] << "\t";
		}
		clustcd16_vaxhigh_vx_file << std::endl;
		clustcd16_vaxhigh_file << std::endl;
	}

	std::string clustcd17_vaxlow_filename;
	clustcd17_vaxlow_filename = s + "_clustcd17_vaxlow.txt";
	std::ofstream clustcd17_vaxlow_file;
	clustcd17_vaxlow_file.open(clustcd17_vaxlow_filename);
	std::string clustcd17_vaxlow_vx_filename;
	clustcd17_vaxlow_vx_filename = s + "_clustcd17_vaxlow_vx.txt";
	std::ofstream clustcd17_vaxlow_vx_file;
	clustcd17_vaxlow_vx_file.open(clustcd17_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd17.size(); i++) {
			clustcd17_vaxlow_vx_file << vaccinations[vaxlowcd17[i]][d] << "\t";
			clustcd17_vaxlow_file << coviddeaths[vaxlowcd17[i]][d] << "\t";
		}
		clustcd17_vaxlow_vx_file << std::endl;
		clustcd17_vaxlow_file << std::endl;
	}
	std::string clustcd17_vaxmed_filename;
	clustcd17_vaxmed_filename = s + "_clustcd17_vaxmed.txt";
	std::ofstream clustcd17_vaxmed_file;
	clustcd17_vaxmed_file.open(clustcd17_vaxmed_filename);
	std::string clustcd17_vaxmed_vx_filename;
	clustcd17_vaxmed_vx_filename = s + "_clustcd17_vaxmed_vx.txt";
	std::ofstream clustcd17_vaxmed_vx_file;
	clustcd17_vaxmed_vx_file.open(clustcd17_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd17.size(); i++) {
			clustcd17_vaxmed_vx_file << vaccinations[vaxmedcd17[i]][d] << "\t";
			clustcd17_vaxmed_file << coviddeaths[vaxmedcd17[i]][d] << "\t";
		}
		clustcd17_vaxmed_vx_file << std::endl;
		clustcd17_vaxmed_file << std::endl;
	}
	std::string clustcd17_vaxhigh_filename;
	clustcd17_vaxhigh_filename = s + "_clustcd17_vaxhigh.txt";
	std::ofstream clustcd17_vaxhigh_file;
	clustcd17_vaxhigh_file.open(clustcd17_vaxhigh_filename);
	std::string clustcd17_vaxhigh_vx_filename;
	clustcd17_vaxhigh_vx_filename = s + "_clustcd17_vaxhigh_vx.txt";
	std::ofstream clustcd17_vaxhigh_vx_file;
	clustcd17_vaxhigh_vx_file.open(clustcd17_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd17.size(); i++) {
			clustcd17_vaxhigh_vx_file << vaccinations[vaxhighcd17[i]][d] << "\t";
			clustcd17_vaxhigh_file << coviddeaths[vaxhighcd17[i]][d] << "\t";
		}
		clustcd17_vaxhigh_vx_file << std::endl;
		clustcd17_vaxhigh_file << std::endl;
	}

	std::string clustcd18_vaxlow_filename;
	clustcd18_vaxlow_filename = s + "_clustcd18_vaxlow.txt";
	std::ofstream clustcd18_vaxlow_file;
	clustcd18_vaxlow_file.open(clustcd18_vaxlow_filename);
	std::string clustcd18_vaxlow_vx_filename;
	clustcd18_vaxlow_vx_filename = s + "_clustcd18_vaxlow_vx.txt";
	std::ofstream clustcd18_vaxlow_vx_file;
	clustcd18_vaxlow_vx_file.open(clustcd18_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd18.size(); i++) {
			clustcd18_vaxlow_vx_file << vaccinations[vaxlowcd18[i]][d] << "\t";
			clustcd18_vaxlow_file << coviddeaths[vaxlowcd18[i]][d] << "\t";
		}
		clustcd18_vaxlow_vx_file << std::endl;
		clustcd18_vaxlow_file << std::endl;
	}
	std::string clustcd18_vaxmed_filename;
	clustcd18_vaxmed_filename = s + "_clustcd18_vaxmed.txt";
	std::ofstream clustcd18_vaxmed_file;
	clustcd18_vaxmed_file.open(clustcd18_vaxmed_filename);
	std::string clustcd18_vaxmed_vx_filename;
	clustcd18_vaxmed_vx_filename = s + "_clustcd18_vaxmed_vx.txt";
	std::ofstream clustcd18_vaxmed_vx_file;
	clustcd18_vaxmed_vx_file.open(clustcd18_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd18.size(); i++) {
			clustcd18_vaxmed_vx_file << vaccinations[vaxmedcd18[i]][d] << "\t";
			clustcd18_vaxmed_file << coviddeaths[vaxmedcd18[i]][d] << "\t";
		}
		clustcd18_vaxmed_vx_file << std::endl;
		clustcd18_vaxmed_file << std::endl;
	}
	std::string clustcd18_vaxhigh_filename;
	clustcd18_vaxhigh_filename = s + "_clustcd18_vaxhigh.txt";
	std::ofstream clustcd18_vaxhigh_file;
	clustcd18_vaxhigh_file.open(clustcd18_vaxhigh_filename);
	std::string clustcd18_vaxhigh_vx_filename;
	clustcd18_vaxhigh_vx_filename = s + "_clustcd18_vaxhigh_vx.txt";
	std::ofstream clustcd18_vaxhigh_vx_file;
	clustcd18_vaxhigh_vx_file.open(clustcd18_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd18.size(); i++) {
			clustcd18_vaxhigh_vx_file << vaccinations[vaxhighcd18[i]][d] << "\t";
			clustcd18_vaxhigh_file << coviddeaths[vaxhighcd18[i]][d] << "\t";
		}
		clustcd18_vaxhigh_vx_file << std::endl;
		clustcd18_vaxhigh_file << std::endl;
	}

	std::string clustcd19_vaxlow_filename;
	clustcd19_vaxlow_filename = s + "_clustcd19_vaxlow.txt";
	std::ofstream clustcd19_vaxlow_file;
	clustcd19_vaxlow_file.open(clustcd19_vaxlow_filename);
	std::string clustcd19_vaxlow_vx_filename;
	clustcd19_vaxlow_vx_filename = s + "_clustcd19_vaxlow_vx.txt";
	std::ofstream clustcd19_vaxlow_vx_file;
	clustcd19_vaxlow_vx_file.open(clustcd19_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowcd19.size(); i++) {
			clustcd19_vaxlow_vx_file << vaccinations[vaxlowcd19[i]][d] << "\t";
			clustcd19_vaxlow_file << coviddeaths[vaxlowcd19[i]][d] << "\t";
		}
		clustcd19_vaxlow_vx_file << std::endl;
		clustcd19_vaxlow_file << std::endl;
	}
	std::string clustcd19_vaxmed_filename;
	clustcd19_vaxmed_filename = s + "_clustcd19_vaxmed.txt";
	std::ofstream clustcd19_vaxmed_file;
	clustcd19_vaxmed_file.open(clustcd19_vaxmed_filename);
	std::string clustcd19_vaxmed_vx_filename;
	clustcd19_vaxmed_vx_filename = s + "_clustcd19_vaxmed_vx.txt";
	std::ofstream clustcd19_vaxmed_vx_file;
	clustcd19_vaxmed_vx_file.open(clustcd19_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedcd19.size(); i++) {
			clustcd19_vaxmed_vx_file << vaccinations[vaxmedcd19[i]][d] << "\t";
			clustcd19_vaxmed_file << coviddeaths[vaxmedcd19[i]][d] << "\t";
		}
		clustcd19_vaxmed_vx_file << std::endl;
		clustcd19_vaxmed_file << std::endl;
	}
	std::string clustcd19_vaxhigh_filename;
	clustcd19_vaxhigh_filename = s + "_clustcd19_vaxhigh.txt";
	std::ofstream clustcd19_vaxhigh_file;
	clustcd19_vaxhigh_file.open(clustcd19_vaxhigh_filename);
	std::string clustcd19_vaxhigh_vx_filename;
	clustcd19_vaxhigh_vx_filename = s + "_clustcd19_vaxhigh_vx.txt";
	std::ofstream clustcd19_vaxhigh_vx_file;
	clustcd19_vaxhigh_vx_file.open(clustcd19_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighcd19.size(); i++) {
			clustcd19_vaxhigh_vx_file << vaccinations[vaxhighcd19[i]][d] << "\t";
			clustcd19_vaxhigh_file << coviddeaths[vaxhighcd19[i]][d] << "\t";
		}
		clustcd19_vaxhigh_vx_file << std::endl;
		clustcd19_vaxhigh_file << std::endl;
	}









	std::string clustem0_vaxlow_filename;
	clustem0_vaxlow_filename = s + "_clustem0_vaxlow.txt";
	std::ofstream clustem0_vaxlow_file;
	clustem0_vaxlow_file.open(clustem0_vaxlow_filename);

	std::string clustem0_vaxlow_vx_filename;
	clustem0_vaxlow_vx_filename = s + "_clustem0_vaxlow_vx.txt";
	std::ofstream clustem0_vaxlow_vx_file;
	clustem0_vaxlow_vx_file.open(clustem0_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem0.size(); i++) {
			clustem0_vaxlow_vx_file << vaccinations[vaxlowem0[i]][d] << "\t";
			clustem0_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem0[i]][d] << "\t";
		}
		clustem0_vaxlow_vx_file << std::endl;
		clustem0_vaxlow_file << std::endl;
	}


	std::string clustem0_vaxmed_filename;
	clustem0_vaxmed_filename = s + "_clustem0_vaxmed.txt";
	std::ofstream clustem0_vaxmed_file;
	clustem0_vaxmed_file.open(clustem0_vaxmed_filename);

	std::string clustem0_vaxmed_vx_filename;
	clustem0_vaxmed_vx_filename = s + "_clustem0_vaxmed_vx.txt";
	std::ofstream clustem0_vaxmed_vx_file;
	clustem0_vaxmed_vx_file.open(clustem0_vaxmed_vx_filename);

	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem0.size(); i++) {
			clustem0_vaxmed_vx_file << vaccinations[vaxmedem0[i]][d] << "\t";
			clustem0_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem0[i]][d] << "\t";
		}
		clustem0_vaxmed_vx_file << std::endl;
		clustem0_vaxmed_file << std::endl;
	}


	std::string clustem0_vaxhigh_filename;
	clustem0_vaxhigh_filename = s + "_clustem0_vaxhigh.txt";
	std::ofstream clustem0_vaxhigh_file;
	clustem0_vaxhigh_file.open(clustem0_vaxhigh_filename);

	std::string clustem0_vaxhigh_vx_filename;
	clustem0_vaxhigh_vx_filename = s + "_clustem0_vaxhigh_vx.txt";
	std::ofstream clustem0_vaxhigh_vx_file;
	clustem0_vaxhigh_vx_file.open(clustem0_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem0.size(); i++) {
			clustem0_vaxhigh_vx_file << vaccinations[vaxhighem0[i]][d] << "\t";
			clustem0_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem0[i]][d] << "\t";
		}
		clustem0_vaxhigh_vx_file << std::endl;
		clustem0_vaxhigh_file << std::endl;
	}


	std::string clustem1_vaxlow_filename;
	clustem1_vaxlow_filename = s + "_clustem1_vaxlow.txt";
	std::ofstream clustem1_vaxlow_file;
	clustem1_vaxlow_file.open(clustem1_vaxlow_filename);

	std::string clustem1_vaxlow_vx_filename;
	clustem1_vaxlow_vx_filename = s + "_clustem1_vaxlow_vx.txt";
	std::ofstream clustem1_vaxlow_vx_file;
	clustem1_vaxlow_vx_file.open(clustem1_vaxlow_vx_filename);

	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem1.size(); i++) {
			clustem1_vaxlow_vx_file << vaccinations[vaxlowem1[i]][d] << "\t";
			clustem1_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem1[i]][d] << "\t";
		}
		clustem1_vaxlow_vx_file << std::endl;
		clustem1_vaxlow_file << std::endl;
	}


	std::string clustem1_vaxmed_filename;
	clustem1_vaxmed_filename = s + "_clustem1_vaxmed.txt";
	std::ofstream clustem1_vaxmed_file;
	clustem1_vaxmed_file.open(clustem1_vaxmed_filename);

	std::string clustem1_vaxmed_vx_filename;
	clustem1_vaxmed_vx_filename = s + "_clustem1_vaxmed_vx.txt";
	std::ofstream clustem1_vaxmed_vx_file;
	clustem1_vaxmed_vx_file.open(clustem1_vaxmed_vx_filename);

	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem1.size(); i++) {
			clustem1_vaxmed_vx_file << vaccinations[vaxmedem1[i]][d] << "\t";
			clustem1_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem1[i]][d] << "\t";
		}
		clustem1_vaxmed_vx_file << std::endl;
		clustem1_vaxmed_file << std::endl;
	}

	std::string clustem1_vaxhigh_filename;
	clustem1_vaxhigh_filename = s + "_clustem1_vaxhigh.txt";
	std::ofstream clustem1_vaxhigh_file;
	clustem1_vaxhigh_file.open(clustem1_vaxhigh_filename);

	std::string clustem1_vaxhigh_vx_filename;
	clustem1_vaxhigh_vx_filename = s + "_clustem1_vaxhigh_vx.txt";
	std::ofstream clustem1_vaxhigh_vx_file;
	clustem1_vaxhigh_vx_file.open(clustem1_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem1.size(); i++) {
			clustem1_vaxhigh_vx_file << vaccinations[vaxhighem1[i]][d] << "\t";
			clustem1_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem1[i]][d] << "\t";
		}
		clustem1_vaxhigh_vx_file << std::endl;
		clustem1_vaxhigh_file << std::endl;
	}

	std::string clustem2_vaxlow_filename;
	clustem2_vaxlow_filename = s + "_clustem2_vaxlow.txt";
	std::ofstream clustem2_vaxlow_file;
	clustem2_vaxlow_file.open(clustem2_vaxlow_filename);
	std::string clustem2_vaxlow_vx_filename;
	clustem2_vaxlow_vx_filename = s + "_clustem2_vaxlow_vx.txt";
	std::ofstream clustem2_vaxlow_vx_file;
	clustem2_vaxlow_vx_file.open(clustem2_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem2.size(); i++) {
			clustem2_vaxlow_vx_file << vaccinations[vaxlowem2[i]][d] << "\t";
			clustem2_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem2[i]][d] << "\t";
		}
		clustem2_vaxlow_vx_file << std::endl;
		clustem2_vaxlow_file << std::endl;
	}
	std::string clustem2_vaxmed_filename;
	clustem2_vaxmed_filename = s + "_clustem2_vaxmed.txt";
	std::ofstream clustem2_vaxmed_file;
	clustem2_vaxmed_file.open(clustem2_vaxmed_filename);
	std::string clustem2_vaxmed_vx_filename;
	clustem2_vaxmed_vx_filename = s + "_clustem2_vaxmed_vx.txt";
	std::ofstream clustem2_vaxmed_vx_file;
	clustem2_vaxmed_vx_file.open(clustem2_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem2.size(); i++) {
			clustem2_vaxmed_vx_file << vaccinations[vaxmedem2[i]][d] << "\t";
			clustem2_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem2[i]][d] << "\t";
		}
		clustem2_vaxmed_vx_file << std::endl;
		clustem2_vaxmed_file << std::endl;
	}
	std::string clustem2_vaxhigh_filename;
	clustem2_vaxhigh_filename = s + "_clustem2_vaxhigh.txt";
	std::ofstream clustem2_vaxhigh_file;
	clustem2_vaxhigh_file.open(clustem2_vaxhigh_filename);
	std::string clustem2_vaxhigh_vx_filename;
	clustem2_vaxhigh_vx_filename = s + "_clustem2_vaxhigh_vx.txt";
	std::ofstream clustem2_vaxhigh_vx_file;
	clustem2_vaxhigh_vx_file.open(clustem2_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem2.size(); i++) {
			clustem2_vaxhigh_vx_file << vaccinations[vaxhighem2[i]][d] << "\t";
			clustem2_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem2[i]][d] << "\t";
		}
		clustem2_vaxhigh_vx_file << std::endl;
		clustem2_vaxhigh_file << std::endl;
	}

	std::string clustem3_vaxlow_filename;
	clustem3_vaxlow_filename = s + "_clustem3_vaxlow.txt";
	std::ofstream clustem3_vaxlow_file;
	clustem3_vaxlow_file.open(clustem3_vaxlow_filename);
	std::string clustem3_vaxlow_vx_filename;
	clustem3_vaxlow_vx_filename = s + "_clustem3_vaxlow_vx.txt";
	std::ofstream clustem3_vaxlow_vx_file;
	clustem3_vaxlow_vx_file.open(clustem3_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem3.size(); i++) {
			clustem3_vaxlow_vx_file << vaccinations[vaxlowem3[i]][d] << "\t";
			clustem3_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem3[i]][d] << "\t";
		}
		clustem3_vaxlow_vx_file << std::endl;
		clustem3_vaxlow_file << std::endl;
	}
	std::string clustem3_vaxmed_filename;
	clustem3_vaxmed_filename = s + "_clustem3_vaxmed.txt";
	std::ofstream clustem3_vaxmed_file;
	clustem3_vaxmed_file.open(clustem3_vaxmed_filename);
	std::string clustem3_vaxmed_vx_filename;
	clustem3_vaxmed_vx_filename = s + "_clustem3_vaxmed_vx.txt";
	std::ofstream clustem3_vaxmed_vx_file;
	clustem3_vaxmed_vx_file.open(clustem3_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem3.size(); i++) {
			clustem3_vaxmed_vx_file << vaccinations[vaxmedem3[i]][d] << "\t";
			clustem3_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem3[i]][d] << "\t";
		}
		clustem3_vaxmed_vx_file << std::endl;
		clustem3_vaxmed_file << std::endl;
	}
	std::string clustem3_vaxhigh_filename;
	clustem3_vaxhigh_filename = s + "_clustem3_vaxhigh.txt";
	std::ofstream clustem3_vaxhigh_file;
	clustem3_vaxhigh_file.open(clustem3_vaxhigh_filename);
	std::string clustem3_vaxhigh_vx_filename;
	clustem3_vaxhigh_vx_filename = s + "_clustem3_vaxhigh_vx.txt";
	std::ofstream clustem3_vaxhigh_vx_file;
	clustem3_vaxhigh_vx_file.open(clustem3_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem3.size(); i++) {
			clustem3_vaxhigh_vx_file << vaccinations[vaxhighem3[i]][d] << "\t";
			clustem3_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem3[i]][d] << "\t";
		}
		clustem3_vaxhigh_vx_file << std::endl;
		clustem3_vaxhigh_file << std::endl;
	}

	std::string clustem4_vaxlow_filename;
	clustem4_vaxlow_filename = s + "_clustem4_vaxlow.txt";
	std::ofstream clustem4_vaxlow_file;
	clustem4_vaxlow_file.open(clustem4_vaxlow_filename);
	std::string clustem4_vaxlow_vx_filename;
	clustem4_vaxlow_vx_filename = s + "_clustem4_vaxlow_vx.txt";
	std::ofstream clustem4_vaxlow_vx_file;
	clustem4_vaxlow_vx_file.open(clustem4_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem4.size(); i++) {
			clustem4_vaxlow_vx_file << vaccinations[vaxlowem4[i]][d] << "\t";
			clustem4_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem4[i]][d] << "\t";
		}
		clustem4_vaxlow_vx_file << std::endl;
		clustem4_vaxlow_file << std::endl;
	}
	std::string clustem4_vaxmed_filename;
	clustem4_vaxmed_filename = s + "_clustem4_vaxmed.txt";
	std::ofstream clustem4_vaxmed_file;
	clustem4_vaxmed_file.open(clustem4_vaxmed_filename);
	std::string clustem4_vaxmed_vx_filename;
	clustem4_vaxmed_vx_filename = s + "_clustem4_vaxmed_vx.txt";
	std::ofstream clustem4_vaxmed_vx_file;
	clustem4_vaxmed_vx_file.open(clustem4_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem4.size(); i++) {
			clustem4_vaxmed_vx_file << vaccinations[vaxmedem4[i]][d] << "\t";
			clustem4_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem4[i]][d] << "\t";
		}
		clustem4_vaxmed_vx_file << std::endl;
		clustem4_vaxmed_file << std::endl;
	}
	std::string clustem4_vaxhigh_filename;
	clustem4_vaxhigh_filename = s + "_clustem4_vaxhigh.txt";
	std::ofstream clustem4_vaxhigh_file;
	clustem4_vaxhigh_file.open(clustem4_vaxhigh_filename);
	std::string clustem4_vaxhigh_vx_filename;
	clustem4_vaxhigh_vx_filename = s + "_clustem4_vaxhigh_vx.txt";
	std::ofstream clustem4_vaxhigh_vx_file;
	clustem4_vaxhigh_vx_file.open(clustem4_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem4.size(); i++) {
			clustem4_vaxhigh_vx_file << vaccinations[vaxhighem4[i]][d] << "\t";
			clustem4_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem4[i]][d] << "\t";
		}
		clustem4_vaxhigh_vx_file << std::endl;
		clustem4_vaxhigh_file << std::endl;
	}

	std::string clustem5_vaxlow_filename;
	clustem5_vaxlow_filename = s + "_clustem5_vaxlow.txt";
	std::ofstream clustem5_vaxlow_file;
	clustem5_vaxlow_file.open(clustem5_vaxlow_filename);
	std::string clustem5_vaxlow_vx_filename;
	clustem5_vaxlow_vx_filename = s + "_clustem5_vaxlow_vx.txt";
	std::ofstream clustem5_vaxlow_vx_file;
	clustem5_vaxlow_vx_file.open(clustem5_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem5.size(); i++) {
			clustem5_vaxlow_vx_file << vaccinations[vaxlowem5[i]][d] << "\t";
			clustem5_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem5[i]][d] << "\t";
		}
		clustem5_vaxlow_vx_file << std::endl;
		clustem5_vaxlow_file << std::endl;
	}
	std::string clustem5_vaxmed_filename;
	clustem5_vaxmed_filename = s + "_clustem5_vaxmed.txt";
	std::ofstream clustem5_vaxmed_file;
	clustem5_vaxmed_file.open(clustem5_vaxmed_filename);
	std::string clustem5_vaxmed_vx_filename;
	clustem5_vaxmed_vx_filename = s + "_clustem5_vaxmed_vx.txt";
	std::ofstream clustem5_vaxmed_vx_file;
	clustem5_vaxmed_vx_file.open(clustem5_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem5.size(); i++) {
			clustem5_vaxmed_vx_file << vaccinations[vaxmedem5[i]][d] << "\t";
			clustem5_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem5[i]][d] << "\t";
		}
		clustem5_vaxmed_vx_file << std::endl;
		clustem5_vaxmed_file << std::endl;
	}
	std::string clustem5_vaxhigh_filename;
	clustem5_vaxhigh_filename = s + "_clustem5_vaxhigh.txt";
	std::ofstream clustem5_vaxhigh_file;
	clustem5_vaxhigh_file.open(clustem5_vaxhigh_filename);
	std::string clustem5_vaxhigh_vx_filename;
	clustem5_vaxhigh_vx_filename = s + "_clustem5_vaxhigh_vx.txt";
	std::ofstream clustem5_vaxhigh_vx_file;
	clustem5_vaxhigh_vx_file.open(clustem5_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem5.size(); i++) {
			clustem5_vaxhigh_vx_file << vaccinations[vaxhighem5[i]][d] << "\t";
			clustem5_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem5[i]][d] << "\t";
		}
		clustem5_vaxhigh_vx_file << std::endl;
		clustem5_vaxhigh_file << std::endl;
	}

	std::string clustem6_vaxlow_filename;
	clustem6_vaxlow_filename = s + "_clustem6_vaxlow.txt";
	std::ofstream clustem6_vaxlow_file;
	clustem6_vaxlow_file.open(clustem6_vaxlow_filename);
	std::string clustem6_vaxlow_vx_filename;
	clustem6_vaxlow_vx_filename = s + "_clustem6_vaxlow_vx.txt";
	std::ofstream clustem6_vaxlow_vx_file;
	clustem6_vaxlow_vx_file.open(clustem6_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem6.size(); i++) {
			clustem6_vaxlow_vx_file << vaccinations[vaxlowem6[i]][d] << "\t";
			clustem6_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem6[i]][d] << "\t";
		}
		clustem6_vaxlow_vx_file << std::endl;
		clustem6_vaxlow_file << std::endl;
	}
	std::string clustem6_vaxmed_filename;
	clustem6_vaxmed_filename = s + "_clustem6_vaxmed.txt";
	std::ofstream clustem6_vaxmed_file;
	clustem6_vaxmed_file.open(clustem6_vaxmed_filename);
	std::string clustem6_vaxmed_vx_filename;
	clustem6_vaxmed_vx_filename = s + "_clustem6_vaxmed_vx.txt";
	std::ofstream clustem6_vaxmed_vx_file;
	clustem6_vaxmed_vx_file.open(clustem6_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem6.size(); i++) {
			clustem6_vaxmed_vx_file << vaccinations[vaxmedem6[i]][d] << "\t";
			clustem6_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem6[i]][d] << "\t";
		}
		clustem6_vaxmed_vx_file << std::endl;
		clustem6_vaxmed_file << std::endl;
	}
	std::string clustem6_vaxhigh_filename;
	clustem6_vaxhigh_filename = s + "_clustem6_vaxhigh.txt";
	std::ofstream clustem6_vaxhigh_file;
	clustem6_vaxhigh_file.open(clustem6_vaxhigh_filename);
	std::string clustem6_vaxhigh_vx_filename;
	clustem6_vaxhigh_vx_filename = s + "_clustem6_vaxhigh_vx.txt";
	std::ofstream clustem6_vaxhigh_vx_file;
	clustem6_vaxhigh_vx_file.open(clustem6_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem6.size(); i++) {
			clustem6_vaxhigh_vx_file << vaccinations[vaxhighem6[i]][d] << "\t";
			clustem6_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem6[i]][d] << "\t";
		}
		clustem6_vaxhigh_vx_file << std::endl;
		clustem6_vaxhigh_file << std::endl;
	}

	std::string clustem7_vaxlow_filename;
	clustem7_vaxlow_filename = s + "_clustem7_vaxlow.txt";
	std::ofstream clustem7_vaxlow_file;
	clustem7_vaxlow_file.open(clustem7_vaxlow_filename);
	std::string clustem7_vaxlow_vx_filename;
	clustem7_vaxlow_vx_filename = s + "_clustem7_vaxlow_vx.txt";
	std::ofstream clustem7_vaxlow_vx_file;
	clustem7_vaxlow_vx_file.open(clustem7_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem7.size(); i++) {
			clustem7_vaxlow_vx_file << vaccinations[vaxlowem7[i]][d] << "\t";
			clustem7_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem7[i]][d] << "\t";
		}
		clustem7_vaxlow_vx_file << std::endl;
		clustem7_vaxlow_file << std::endl;
	}
	std::string clustem7_vaxmed_filename;
	clustem7_vaxmed_filename = s + "_clustem7_vaxmed.txt";
	std::ofstream clustem7_vaxmed_file;
	clustem7_vaxmed_file.open(clustem7_vaxmed_filename);
	std::string clustem7_vaxmed_vx_filename;
	clustem7_vaxmed_vx_filename = s + "_clustem7_vaxmed_vx.txt";
	std::ofstream clustem7_vaxmed_vx_file;
	clustem7_vaxmed_vx_file.open(clustem7_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem7.size(); i++) {
			clustem7_vaxmed_vx_file << vaccinations[vaxmedem7[i]][d] << "\t";
			clustem7_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem7[i]][d] << "\t";
		}
		clustem7_vaxmed_vx_file << std::endl;
		clustem7_vaxmed_file << std::endl;
	}
	std::string clustem7_vaxhigh_filename;
	clustem7_vaxhigh_filename = s + "_clustem7_vaxhigh.txt";
	std::ofstream clustem7_vaxhigh_file;
	clustem7_vaxhigh_file.open(clustem7_vaxhigh_filename);
	std::string clustem7_vaxhigh_vx_filename;
	clustem7_vaxhigh_vx_filename = s + "_clustem7_vaxhigh_vx.txt";
	std::ofstream clustem7_vaxhigh_vx_file;
	clustem7_vaxhigh_vx_file.open(clustem7_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem7.size(); i++) {
			clustem7_vaxhigh_vx_file << vaccinations[vaxhighem7[i]][d] << "\t";
			clustem7_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem7[i]][d] << "\t";
		}
		clustem7_vaxhigh_vx_file << std::endl;
		clustem7_vaxhigh_file << std::endl;
	}

	std::string clustem8_vaxlow_filename;
	clustem8_vaxlow_filename = s + "_clustem8_vaxlow.txt";
	std::ofstream clustem8_vaxlow_file;
	clustem8_vaxlow_file.open(clustem8_vaxlow_filename);
	std::string clustem8_vaxlow_vx_filename;
	clustem8_vaxlow_vx_filename = s + "_clustem8_vaxlow_vx.txt";
	std::ofstream clustem8_vaxlow_vx_file;
	clustem8_vaxlow_vx_file.open(clustem8_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem8.size(); i++) {
			clustem8_vaxlow_vx_file << vaccinations[vaxlowem8[i]][d] << "\t";
			clustem8_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem8[i]][d] << "\t";
		}
		clustem8_vaxlow_vx_file << std::endl;
		clustem8_vaxlow_file << std::endl;
	}
	std::string clustem8_vaxmed_filename;
	clustem8_vaxmed_filename = s + "_clustem8_vaxmed.txt";
	std::ofstream clustem8_vaxmed_file;
	clustem8_vaxmed_file.open(clustem8_vaxmed_filename);
	std::string clustem8_vaxmed_vx_filename;
	clustem8_vaxmed_vx_filename = s + "_clustem8_vaxmed_vx.txt";
	std::ofstream clustem8_vaxmed_vx_file;
	clustem8_vaxmed_vx_file.open(clustem8_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem8.size(); i++) {
			clustem8_vaxmed_vx_file << vaccinations[vaxmedem8[i]][d] << "\t";
			clustem8_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem8[i]][d] << "\t";
		}
		clustem8_vaxmed_vx_file << std::endl;
		clustem8_vaxmed_file << std::endl;
	}
	std::string clustem8_vaxhigh_filename;
	clustem8_vaxhigh_filename = s + "_clustem8_vaxhigh.txt";
	std::ofstream clustem8_vaxhigh_file;
	clustem8_vaxhigh_file.open(clustem8_vaxhigh_filename);
	std::string clustem8_vaxhigh_vx_filename;
	clustem8_vaxhigh_vx_filename = s + "_clustem8_vaxhigh_vx.txt";
	std::ofstream clustem8_vaxhigh_vx_file;
	clustem8_vaxhigh_vx_file.open(clustem8_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem8.size(); i++) {
			clustem8_vaxhigh_vx_file << vaccinations[vaxhighem8[i]][d] << "\t";
			clustem8_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem8[i]][d] << "\t";
		}
		clustem8_vaxhigh_vx_file << std::endl;
		clustem8_vaxhigh_file << std::endl;
	}


	std::string clustem9_vaxlow_filename;
	clustem9_vaxlow_filename = s + "_clustem9_vaxlow.txt";
	std::ofstream clustem9_vaxlow_file;
	clustem9_vaxlow_file.open(clustem9_vaxlow_filename);
	std::string clustem9_vaxlow_vx_filename;
	clustem9_vaxlow_vx_filename = s + "_clustem9_vaxlow_vx.txt";
	std::ofstream clustem9_vaxlow_vx_file;
	clustem9_vaxlow_vx_file.open(clustem9_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem9.size(); i++) {
			clustem9_vaxlow_vx_file << vaccinations[vaxlowem9[i]][d] << "\t";
			clustem9_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem9[i]][d] << "\t";
		}
		clustem9_vaxlow_vx_file << std::endl;
		clustem9_vaxlow_file << std::endl;
	}
	std::string clustem9_vaxmed_filename;
	clustem9_vaxmed_filename = s + "_clustem9_vaxmed.txt";
	std::ofstream clustem9_vaxmed_file;
	clustem9_vaxmed_file.open(clustem9_vaxmed_filename);
	std::string clustem9_vaxmed_vx_filename;
	clustem9_vaxmed_vx_filename = s + "_clustem9_vaxmed_vx.txt";
	std::ofstream clustem9_vaxmed_vx_file;
	clustem9_vaxmed_vx_file.open(clustem9_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem9.size(); i++) {
			clustem9_vaxmed_vx_file << vaccinations[vaxmedem9[i]][d] << "\t";
			clustem9_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem9[i]][d] << "\t";
		}
		clustem9_vaxmed_vx_file << std::endl;
		clustem9_vaxmed_file << std::endl;
	}
	std::string clustem9_vaxhigh_filename;
	clustem9_vaxhigh_filename = s + "_clustem9_vaxhigh.txt";
	std::ofstream clustem9_vaxhigh_file;
	clustem9_vaxhigh_file.open(clustem9_vaxhigh_filename);
	std::string clustem9_vaxhigh_vx_filename;
	clustem9_vaxhigh_vx_filename = s + "_clustem9_vaxhigh_vx.txt";
	std::ofstream clustem9_vaxhigh_vx_file;
	clustem9_vaxhigh_vx_file.open(clustem9_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem9.size(); i++) {
			clustem9_vaxhigh_vx_file << vaccinations[vaxhighem9[i]][d] << "\t";
			clustem9_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem9[i]][d] << "\t";
		}
		clustem9_vaxhigh_vx_file << std::endl;
		clustem9_vaxhigh_file << std::endl;
	}

	std::string clustem10_vaxlow_filename;
	clustem10_vaxlow_filename = s + "_clustem10_vaxlow.txt";
	std::ofstream clustem10_vaxlow_file;
	clustem10_vaxlow_file.open(clustem10_vaxlow_filename);
	std::string clustem10_vaxlow_vx_filename;
	clustem10_vaxlow_vx_filename = s + "_clustem10_vaxlow_vx.txt";
	std::ofstream clustem10_vaxlow_vx_file;
	clustem10_vaxlow_vx_file.open(clustem10_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem10.size(); i++) {
			clustem10_vaxlow_vx_file << vaccinations[vaxlowem10[i]][d] << "\t";
			clustem10_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem10[i]][d] << "\t";
		}
		clustem10_vaxlow_vx_file << std::endl;
		clustem10_vaxlow_file << std::endl;
	}
	std::string clustem10_vaxmed_filename;
	clustem10_vaxmed_filename = s + "_clustem10_vaxmed.txt";
	std::ofstream clustem10_vaxmed_file;
	clustem10_vaxmed_file.open(clustem10_vaxmed_filename);
	std::string clustem10_vaxmed_vx_filename;
	clustem10_vaxmed_vx_filename = s + "_clustem10_vaxmed_vx.txt";
	std::ofstream clustem10_vaxmed_vx_file;
	clustem10_vaxmed_vx_file.open(clustem10_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem10.size(); i++) {
			clustem10_vaxmed_vx_file << vaccinations[vaxmedem10[i]][d] << "\t";
			clustem10_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem10[i]][d] << "\t";
		}
		clustem10_vaxmed_vx_file << std::endl;
		clustem10_vaxmed_file << std::endl;
	}
	std::string clustem10_vaxhigh_filename;
	clustem10_vaxhigh_filename = s + "_clustem10_vaxhigh.txt";
	std::ofstream clustem10_vaxhigh_file;
	clustem10_vaxhigh_file.open(clustem10_vaxhigh_filename);
	std::string clustem10_vaxhigh_vx_filename;
	clustem10_vaxhigh_vx_filename = s + "_clustem10_vaxhigh_vx.txt";
	std::ofstream clustem10_vaxhigh_vx_file;
	clustem10_vaxhigh_vx_file.open(clustem10_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem10.size(); i++) {
			clustem10_vaxhigh_vx_file << vaccinations[vaxhighem10[i]][d] << "\t";
			clustem10_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem10[i]][d] << "\t";
		}
		clustem10_vaxhigh_vx_file << std::endl;
		clustem10_vaxhigh_file << std::endl;
	}

	std::string clustem11_vaxlow_filename;
	clustem11_vaxlow_filename = s + "_clustem11_vaxlow.txt";
	std::ofstream clustem11_vaxlow_file;
	clustem11_vaxlow_file.open(clustem11_vaxlow_filename);
	std::string clustem11_vaxlow_vx_filename;
	clustem11_vaxlow_vx_filename = s + "_clustem11_vaxlow_vx.txt";
	std::ofstream clustem11_vaxlow_vx_file;
	clustem11_vaxlow_vx_file.open(clustem11_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem11.size(); i++) {
			clustem11_vaxlow_vx_file << vaccinations[vaxlowem11[i]][d] << "\t";
			clustem11_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem11[i]][d] << "\t";
		}
		clustem11_vaxlow_vx_file << std::endl;
		clustem11_vaxlow_file << std::endl;
	}
	std::string clustem11_vaxmed_filename;
	clustem11_vaxmed_filename = s + "_clustem11_vaxmed.txt";
	std::ofstream clustem11_vaxmed_file;
	clustem11_vaxmed_file.open(clustem11_vaxmed_filename);
	std::string clustem11_vaxmed_vx_filename;
	clustem11_vaxmed_vx_filename = s + "_clustem11_vaxmed_vx.txt";
	std::ofstream clustem11_vaxmed_vx_file;
	clustem11_vaxmed_vx_file.open(clustem11_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem11.size(); i++) {
			clustem11_vaxmed_vx_file << vaccinations[vaxmedem11[i]][d] << "\t";
			clustem11_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem11[i]][d] << "\t";
		}
		clustem11_vaxmed_vx_file << std::endl;
		clustem11_vaxmed_file << std::endl;
	}
	std::string clustem11_vaxhigh_filename;
	clustem11_vaxhigh_filename = s + "_clustem11_vaxhigh.txt";
	std::ofstream clustem11_vaxhigh_file;
	clustem11_vaxhigh_file.open(clustem11_vaxhigh_filename);
	std::string clustem11_vaxhigh_vx_filename;
	clustem11_vaxhigh_vx_filename = s + "_clustem11_vaxhigh_vx.txt";
	std::ofstream clustem11_vaxhigh_vx_file;
	clustem11_vaxhigh_vx_file.open(clustem11_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem11.size(); i++) {
			clustem11_vaxhigh_vx_file << vaccinations[vaxhighem11[i]][d] << "\t";
			clustem11_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem11[i]][d] << "\t";
		}
		clustem11_vaxhigh_vx_file << std::endl;
		clustem11_vaxhigh_file << std::endl;
	}

	std::string clustem12_vaxlow_filename;
	clustem12_vaxlow_filename = s + "_clustem12_vaxlow.txt";
	std::ofstream clustem12_vaxlow_file;
	clustem12_vaxlow_file.open(clustem12_vaxlow_filename);
	std::string clustem12_vaxlow_vx_filename;
	clustem12_vaxlow_vx_filename = s + "_clustem12_vaxlow_vx.txt";
	std::ofstream clustem12_vaxlow_vx_file;
	clustem12_vaxlow_vx_file.open(clustem12_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem12.size(); i++) {
			clustem12_vaxlow_vx_file << vaccinations[vaxlowem12[i]][d] << "\t";
			clustem12_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem12[i]][d] << "\t";
		}
		clustem12_vaxlow_vx_file << std::endl;
		clustem12_vaxlow_file << std::endl;
	}
	std::string clustem12_vaxmed_filename;
	clustem12_vaxmed_filename = s + "_clustem12_vaxmed.txt";
	std::ofstream clustem12_vaxmed_file;
	clustem12_vaxmed_file.open(clustem12_vaxmed_filename);
	std::string clustem12_vaxmed_vx_filename;
	clustem12_vaxmed_vx_filename = s + "_clustem12_vaxmed_vx.txt";
	std::ofstream clustem12_vaxmed_vx_file;
	clustem12_vaxmed_vx_file.open(clustem12_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem12.size(); i++) {
			clustem12_vaxmed_vx_file << vaccinations[vaxmedem12[i]][d] << "\t";
			clustem12_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem12[i]][d] << "\t";
		}
		clustem12_vaxmed_vx_file << std::endl;
		clustem12_vaxmed_file << std::endl;
	}
	std::string clustem12_vaxhigh_filename;
	clustem12_vaxhigh_filename = s + "_clustem12_vaxhigh.txt";
	std::ofstream clustem12_vaxhigh_file;
	clustem12_vaxhigh_file.open(clustem12_vaxhigh_filename);
	std::string clustem12_vaxhigh_vx_filename;
	clustem12_vaxhigh_vx_filename = s + "_clustem12_vaxhigh_vx.txt";
	std::ofstream clustem12_vaxhigh_vx_file;
	clustem12_vaxhigh_vx_file.open(clustem12_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem12.size(); i++) {
			clustem12_vaxhigh_vx_file << vaccinations[vaxhighem12[i]][d] << "\t";
			clustem12_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem12[i]][d] << "\t";
		}
		clustem12_vaxhigh_vx_file << std::endl;
		clustem12_vaxhigh_file << std::endl;
	}

	std::string clustem13_vaxlow_filename;
	clustem13_vaxlow_filename = s + "_clustem13_vaxlow.txt";
	std::ofstream clustem13_vaxlow_file;
	clustem13_vaxlow_file.open(clustem13_vaxlow_filename);
	std::string clustem13_vaxlow_vx_filename;
	clustem13_vaxlow_vx_filename = s + "_clustem13_vaxlow_vx.txt";
	std::ofstream clustem13_vaxlow_vx_file;
	clustem13_vaxlow_vx_file.open(clustem13_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem13.size(); i++) {
			clustem13_vaxlow_vx_file << vaccinations[vaxlowem13[i]][d] << "\t";
			clustem13_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem13[i]][d] << "\t";
		}
		clustem13_vaxlow_vx_file << std::endl;
		clustem13_vaxlow_file << std::endl;
	}
	std::string clustem13_vaxmed_filename;
	clustem13_vaxmed_filename = s + "_clustem13_vaxmed.txt";
	std::ofstream clustem13_vaxmed_file;
	clustem13_vaxmed_file.open(clustem13_vaxmed_filename);
	std::string clustem13_vaxmed_vx_filename;
	clustem13_vaxmed_vx_filename = s + "_clustem13_vaxmed_vx.txt";
	std::ofstream clustem13_vaxmed_vx_file;
	clustem13_vaxmed_vx_file.open(clustem13_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem13.size(); i++) {
			clustem13_vaxmed_vx_file << vaccinations[vaxmedem13[i]][d] << "\t";
			clustem13_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem13[i]][d] << "\t";
		}
		clustem13_vaxmed_vx_file << std::endl;
		clustem13_vaxmed_file << std::endl;
	}
	std::string clustem13_vaxhigh_filename;
	clustem13_vaxhigh_filename = s + "_clustem13_vaxhigh.txt";
	std::ofstream clustem13_vaxhigh_file;
	clustem13_vaxhigh_file.open(clustem13_vaxhigh_filename);
	std::string clustem13_vaxhigh_vx_filename;
	clustem13_vaxhigh_vx_filename = s + "_clustem13_vaxhigh_vx.txt";
	std::ofstream clustem13_vaxhigh_vx_file;
	clustem13_vaxhigh_vx_file.open(clustem13_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem13.size(); i++) {
			clustem13_vaxhigh_vx_file << vaccinations[vaxhighem13[i]][d] << "\t";
			clustem13_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem13[i]][d] << "\t";
		}
		clustem13_vaxhigh_vx_file << std::endl;
		clustem13_vaxhigh_file << std::endl;
	}



	std::string clustem14_vaxlow_filename;
	clustem14_vaxlow_filename = s + "_clustem14_vaxlow.txt";
	std::ofstream clustem14_vaxlow_file;
	clustem14_vaxlow_file.open(clustem14_vaxlow_filename);
	std::string clustem14_vaxlow_vx_filename;
	clustem14_vaxlow_vx_filename = s + "_clustem14_vaxlow_vx.txt";
	std::ofstream clustem14_vaxlow_vx_file;
	clustem14_vaxlow_vx_file.open(clustem14_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem14.size(); i++) {
			clustem14_vaxlow_vx_file << vaccinations[vaxlowem14[i]][d] << "\t";
			clustem14_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem14[i]][d] << "\t";
		}
		clustem14_vaxlow_vx_file << std::endl;
		clustem14_vaxlow_file << std::endl;
	}
	std::string clustem14_vaxmed_filename;
	clustem14_vaxmed_filename = s + "_clustem14_vaxmed.txt";
	std::ofstream clustem14_vaxmed_file;
	clustem14_vaxmed_file.open(clustem14_vaxmed_filename);
	std::string clustem14_vaxmed_vx_filename;
	clustem14_vaxmed_vx_filename = s + "_clustem14_vaxmed_vx.txt";
	std::ofstream clustem14_vaxmed_vx_file;
	clustem14_vaxmed_vx_file.open(clustem14_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem14.size(); i++) {
			clustem14_vaxmed_vx_file << vaccinations[vaxmedem14[i]][d] << "\t";
			clustem14_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem14[i]][d] << "\t";
		}
		clustem14_vaxmed_vx_file << std::endl;
		clustem14_vaxmed_file << std::endl;
	}
	std::string clustem14_vaxhigh_filename;
	clustem14_vaxhigh_filename = s + "_clustem14_vaxhigh.txt";
	std::ofstream clustem14_vaxhigh_file;
	clustem14_vaxhigh_file.open(clustem14_vaxhigh_filename);
	std::string clustem14_vaxhigh_vx_filename;
	clustem14_vaxhigh_vx_filename = s + "_clustem14_vaxhigh_vx.txt";
	std::ofstream clustem14_vaxhigh_vx_file;
	clustem14_vaxhigh_vx_file.open(clustem14_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem14.size(); i++) {
			clustem14_vaxhigh_vx_file << vaccinations[vaxhighem14[i]][d] << "\t";
			clustem14_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem14[i]][d] << "\t";
		}
		clustem14_vaxhigh_vx_file << std::endl;
		clustem14_vaxhigh_file << std::endl;
	}



	std::string clustem15_vaxlow_filename;
	clustem15_vaxlow_filename = s + "_clustem15_vaxlow.txt";
	std::ofstream clustem15_vaxlow_file;
	clustem15_vaxlow_file.open(clustem15_vaxlow_filename);
	std::string clustem15_vaxlow_vx_filename;
	clustem15_vaxlow_vx_filename = s + "_clustem15_vaxlow_vx.txt";
	std::ofstream clustem15_vaxlow_vx_file;
	clustem15_vaxlow_vx_file.open(clustem15_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem15.size(); i++) {
			clustem15_vaxlow_vx_file << vaccinations[vaxlowem15[i]][d] << "\t";
			clustem15_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem15[i]][d] << "\t";
		}
		clustem15_vaxlow_vx_file << std::endl;
		clustem15_vaxlow_file << std::endl;
	}
	std::string clustem15_vaxmed_filename;
	clustem15_vaxmed_filename = s + "_clustem15_vaxmed.txt";
	std::ofstream clustem15_vaxmed_file;
	clustem15_vaxmed_file.open(clustem15_vaxmed_filename);
	std::string clustem15_vaxmed_vx_filename;
	clustem15_vaxmed_vx_filename = s + "_clustem15_vaxmed_vx.txt";
	std::ofstream clustem15_vaxmed_vx_file;
	clustem15_vaxmed_vx_file.open(clustem15_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem15.size(); i++) {
			clustem15_vaxmed_vx_file << vaccinations[vaxmedem15[i]][d] << "\t";
			clustem15_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem15[i]][d] << "\t";
		}
		clustem15_vaxmed_vx_file << std::endl;
		clustem15_vaxmed_file << std::endl;
	}
	std::string clustem15_vaxhigh_filename;
	clustem15_vaxhigh_filename = s + "_clustem15_vaxhigh.txt";
	std::ofstream clustem15_vaxhigh_file;
	clustem15_vaxhigh_file.open(clustem15_vaxhigh_filename);
	std::string clustem15_vaxhigh_vx_filename;
	clustem15_vaxhigh_vx_filename = s + "_clustem15_vaxhigh_vx.txt";
	std::ofstream clustem15_vaxhigh_vx_file;
	clustem15_vaxhigh_vx_file.open(clustem15_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem15.size(); i++) {
			clustem15_vaxhigh_vx_file << vaccinations[vaxhighem15[i]][d] << "\t";
			clustem15_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem15[i]][d] << "\t";
		}
		clustem15_vaxhigh_vx_file << std::endl;
		clustem15_vaxhigh_file << std::endl;
	}



	std::string clustem16_vaxlow_filename;
	clustem16_vaxlow_filename = s + "_clustem16_vaxlow.txt";
	std::ofstream clustem16_vaxlow_file;
	clustem16_vaxlow_file.open(clustem16_vaxlow_filename);
	std::string clustem16_vaxlow_vx_filename;
	clustem16_vaxlow_vx_filename = s + "_clustem16_vaxlow_vx.txt";
	std::ofstream clustem16_vaxlow_vx_file;
	clustem16_vaxlow_vx_file.open(clustem16_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem16.size(); i++) {
			clustem16_vaxlow_vx_file << vaccinations[vaxlowem16[i]][d] << "\t";
			clustem16_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem16[i]][d] << "\t";
		}
		clustem16_vaxlow_vx_file << std::endl;
		clustem16_vaxlow_file << std::endl;
	}
	std::string clustem16_vaxmed_filename;
	clustem16_vaxmed_filename = s + "_clustem16_vaxmed.txt";
	std::ofstream clustem16_vaxmed_file;
	clustem16_vaxmed_file.open(clustem16_vaxmed_filename);
	std::string clustem16_vaxmed_vx_filename;
	clustem16_vaxmed_vx_filename = s + "_clustem16_vaxmed_vx.txt";
	std::ofstream clustem16_vaxmed_vx_file;
	clustem16_vaxmed_vx_file.open(clustem16_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem16.size(); i++) {
			clustem16_vaxmed_vx_file << vaccinations[vaxmedem16[i]][d] << "\t";
			clustem16_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem16[i]][d] << "\t";
		}
		clustem16_vaxmed_vx_file << std::endl;
		clustem16_vaxmed_file << std::endl;
	}
	std::string clustem16_vaxhigh_filename;
	clustem16_vaxhigh_filename = s + "_clustem16_vaxhigh.txt";
	std::ofstream clustem16_vaxhigh_file;
	clustem16_vaxhigh_file.open(clustem16_vaxhigh_filename);
	std::string clustem16_vaxhigh_vx_filename;
	clustem16_vaxhigh_vx_filename = s + "_clustem16_vaxhigh_vx.txt";
	std::ofstream clustem16_vaxhigh_vx_file;
	clustem16_vaxhigh_vx_file.open(clustem16_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem16.size(); i++) {
			clustem16_vaxhigh_vx_file << vaccinations[vaxhighem16[i]][d] << "\t";
			clustem16_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem16[i]][d] << "\t";
		}
		clustem16_vaxhigh_vx_file << std::endl;
		clustem16_vaxhigh_file << std::endl;
	}




	std::string clustem17_vaxlow_filename;
	clustem17_vaxlow_filename = s + "_clustem17_vaxlow.txt";
	std::ofstream clustem17_vaxlow_file;
	clustem17_vaxlow_file.open(clustem17_vaxlow_filename);
	std::string clustem17_vaxlow_vx_filename;
	clustem17_vaxlow_vx_filename = s + "_clustem17_vaxlow_vx.txt";
	std::ofstream clustem17_vaxlow_vx_file;
	clustem17_vaxlow_vx_file.open(clustem17_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem17.size(); i++) {
			clustem17_vaxlow_vx_file << vaccinations[vaxlowem17[i]][d] << "\t";
			clustem17_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem17[i]][d] << "\t";
		}
		clustem17_vaxlow_vx_file << std::endl;
		clustem17_vaxlow_file << std::endl;
	}
	std::string clustem17_vaxmed_filename;
	clustem17_vaxmed_filename = s + "_clustem17_vaxmed.txt";
	std::ofstream clustem17_vaxmed_file;
	clustem17_vaxmed_file.open(clustem17_vaxmed_filename);
	std::string clustem17_vaxmed_vx_filename;
	clustem17_vaxmed_vx_filename = s + "_clustem17_vaxmed_vx.txt";
	std::ofstream clustem17_vaxmed_vx_file;
	clustem17_vaxmed_vx_file.open(clustem17_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem17.size(); i++) {
			clustem17_vaxmed_vx_file << vaccinations[vaxmedem17[i]][d] << "\t";
			clustem17_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem17[i]][d] << "\t";
		}
		clustem17_vaxmed_vx_file << std::endl;
		clustem17_vaxmed_file << std::endl;
	}
	std::string clustem17_vaxhigh_filename;
	clustem17_vaxhigh_filename = s + "_clustem17_vaxhigh.txt";
	std::ofstream clustem17_vaxhigh_file;
	clustem17_vaxhigh_file.open(clustem17_vaxhigh_filename);
	std::string clustem17_vaxhigh_vx_filename;
	clustem17_vaxhigh_vx_filename = s + "_clustem17_vaxhigh_vx.txt";
	std::ofstream clustem17_vaxhigh_vx_file;
	clustem17_vaxhigh_vx_file.open(clustem17_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem17.size(); i++) {
			clustem17_vaxhigh_vx_file << vaccinations[vaxhighem17[i]][d] << "\t";
			clustem17_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem17[i]][d] << "\t";
		}
		clustem17_vaxhigh_vx_file << std::endl;
		clustem17_vaxhigh_file << std::endl;
	}



	std::string clustem18_vaxlow_filename;
	clustem18_vaxlow_filename = s + "_clustem18_vaxlow.txt";
	std::ofstream clustem18_vaxlow_file;
	clustem18_vaxlow_file.open(clustem18_vaxlow_filename);
	std::string clustem18_vaxlow_vx_filename;
	clustem18_vaxlow_vx_filename = s + "_clustem18_vaxlow_vx.txt";
	std::ofstream clustem18_vaxlow_vx_file;
	clustem18_vaxlow_vx_file.open(clustem18_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem18.size(); i++) {
			clustem18_vaxlow_vx_file << vaccinations[vaxlowem18[i]][d] << "\t";
			clustem18_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem18[i]][d] << "\t";
		}
		clustem18_vaxlow_vx_file << std::endl;
		clustem18_vaxlow_file << std::endl;
	}
	std::string clustem18_vaxmed_filename;
	clustem18_vaxmed_filename = s + "_clustem18_vaxmed.txt";
	std::ofstream clustem18_vaxmed_file;
	clustem18_vaxmed_file.open(clustem18_vaxmed_filename);
	std::string clustem18_vaxmed_vx_filename;
	clustem18_vaxmed_vx_filename = s + "_clustem18_vaxmed_vx.txt";
	std::ofstream clustem18_vaxmed_vx_file;
	clustem18_vaxmed_vx_file.open(clustem18_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem18.size(); i++) {
			clustem18_vaxmed_vx_file << vaccinations[vaxmedem18[i]][d] << "\t";
			clustem18_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem18[i]][d] << "\t";
		}
		clustem18_vaxmed_vx_file << std::endl;
		clustem18_vaxmed_file << std::endl;
	}
	std::string clustem18_vaxhigh_filename;
	clustem18_vaxhigh_filename = s + "_clustem18_vaxhigh.txt";
	std::ofstream clustem18_vaxhigh_file;
	clustem18_vaxhigh_file.open(clustem18_vaxhigh_filename);
	std::string clustem18_vaxhigh_vx_filename;
	clustem18_vaxhigh_vx_filename = s + "_clustem18_vaxhigh_vx.txt";
	std::ofstream clustem18_vaxhigh_vx_file;
	clustem18_vaxhigh_vx_file.open(clustem18_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem18.size(); i++) {
			clustem18_vaxhigh_vx_file << vaccinations[vaxhighem18[i]][d] << "\t";
			clustem18_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem18[i]][d] << "\t";
		}
		clustem18_vaxhigh_vx_file << std::endl;
		clustem18_vaxhigh_file << std::endl;
	}


	std::string clustem19_vaxlow_filename;
	clustem19_vaxlow_filename = s + "_clustem19_vaxlow.txt";
	std::ofstream clustem19_vaxlow_file;
	clustem19_vaxlow_file.open(clustem19_vaxlow_filename);
	std::string clustem19_vaxlow_vx_filename;
	clustem19_vaxlow_vx_filename = s + "_clustem19_vaxlow_vx.txt";
	std::ofstream clustem19_vaxlow_vx_file;
	clustem19_vaxlow_vx_file.open(clustem19_vaxlow_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxlowem19.size(); i++) {
			clustem19_vaxlow_vx_file << vaccinations[vaxlowem19[i]][d] << "\t";
			clustem19_vaxlow_file << excessmortalitypscoreinterpolated[vaxlowem19[i]][d] << "\t";
		}
		clustem19_vaxlow_vx_file << std::endl;
		clustem19_vaxlow_file << std::endl;
	}
	std::string clustem19_vaxmed_filename;
	clustem19_vaxmed_filename = s + "_clustem19_vaxmed.txt";
	std::ofstream clustem19_vaxmed_file;
	clustem19_vaxmed_file.open(clustem19_vaxmed_filename);
	std::string clustem19_vaxmed_vx_filename;
	clustem19_vaxmed_vx_filename = s + "_clustem19_vaxmed_vx.txt";
	std::ofstream clustem19_vaxmed_vx_file;
	clustem19_vaxmed_vx_file.open(clustem19_vaxmed_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxmedem19.size(); i++) {
			clustem19_vaxmed_vx_file << vaccinations[vaxmedem19[i]][d] << "\t";
			clustem19_vaxmed_file << excessmortalitypscoreinterpolated[vaxmedem19[i]][d] << "\t";
		}
		clustem19_vaxmed_vx_file << std::endl;
		clustem19_vaxmed_file << std::endl;
	}
	std::string clustem19_vaxhigh_filename;
	clustem19_vaxhigh_filename = s + "_clustem19_vaxhigh.txt";
	std::ofstream clustem19_vaxhigh_file;
	clustem19_vaxhigh_file.open(clustem19_vaxhigh_filename);
	std::string clustem19_vaxhigh_vx_filename;
	clustem19_vaxhigh_vx_filename = s + "_clustem19_vaxhigh_vx.txt";
	std::ofstream clustem19_vaxhigh_vx_file;
	clustem19_vaxhigh_vx_file.open(clustem19_vaxhigh_vx_filename);
	for (unsigned int d = 0; d < numdates; d++) {
		for (unsigned int i = 0; i < vaxhighem19.size(); i++) {
			clustem19_vaxhigh_vx_file << vaccinations[vaxhighem19[i]][d] << "\t";
			clustem19_vaxhigh_file << excessmortalitypscoreinterpolated[vaxhighem19[i]][d] << "\t";
		}
		clustem19_vaxhigh_vx_file << std::endl;
		clustem19_vaxhigh_file << std::endl;
	}


	return 0;

}





int main()
{
	int numprevaxdates = 342;

	auto start = std::chrono::high_resolution_clock::now();

	using namespace boost::gregorian;

	std::string outputfilename;
	outputfilename = "output.txt";
	std::ofstream outputfile;
	outputfile.open(outputfilename);

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
	std::ifstream coviddeathsfile;
	coviddeathsfile.open(coviddeathsfilename);
	if (coviddeathsfile.fail()) {
		std::cout << "Failed to open " << coviddeathsfilename << std::endl;
		std::cout << "Press Enter";
		std::cin.get();
		return 0;
	}
	while (!coviddeathsfile.eof()) {
		getline(coviddeathsfile, str); // first row is header, so ignore
		getline(coviddeathsfile, str); // get first row of data
		boost::split(fields, str, boost::is_any_of(","));
		code.push_back(fields[0]); // add first country
		continent.push_back(fields[1]);
		entity.push_back(fields[2]);
		if (fields[1] != "")
			numcountries++;

		previouscountry = fields[2];
		firstdate = from_simple_string(fields[3]);
		lastdate = from_simple_string(fields[3]);
		while (getline(coviddeathsfile, str)) {
			if (!str.empty()) {
				try {
					boost::split(fields, str, boost::is_any_of(","));
					if (!fields[0].empty() && (std::find(code.begin(), code.end(), fields[0]) == code.end())) { // if country not in list, add to list
						code.push_back(fields[0]);
						continent.push_back(fields[1]);
						entity.push_back(fields[2]);
						if (fields[1] != "")
							numcountries++;
					}
					newdate = from_simple_string(fields[3]);
					if (newdate < firstdate)
						firstdate = newdate;
					if (newdate > lastdate)
						lastdate = newdate;
				}
				catch (const std::exception& e) {
					std::cout << e.what();
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
	std::ofstream datesfile;
	datesfile.open(datesfilename);
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


	std::vector<std::vector<double>> coviddeaths(numentities, std::vector<double>(numdates, nan("")));
	std::vector<std::vector<double>> vaccinations(numentities, std::vector<double>(numdates, nan("")));

		



	days dd;
	int dateint;






	///////////////
	// LOCKDOWNS //
	///////////////
	std::string lockdownfilename;
	lockdownfilename = "owid-covid-data.csv";
	std::ifstream lockdownfile;
	std::vector<std::vector<double>> lockdown(numentities, std::vector<double>(numdates, nan("")));

	lockdownfile.open(lockdownfilename);
	while (!lockdownfile.eof()) {
		getline(lockdownfile, str); // first row is header, so ignore
		while (getline(lockdownfile, str)) {
			if (!str.empty()) {
				try {
					boost::split(fields, str, boost::is_any_of(","));
					d = from_simple_string(fields[3]);
					dd = d - firstdate;
					dateint = dd.days();
					if (!fields[0].empty() && !fields[47].empty() && (dateint < numdates)) {
						lockdown[CountryIndex(code, fields[0])][dateint] = stod(fields[47]);
					}
				}
				catch (const std::exception& e) {
					std::cout << e.what();
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


	std::vector<double> cumlockdowns(numentities);
	// Afghanistan,AFG,2020-01-21,0
	while (!lockdownfile.eof()) {
		getline(lockdownfile, str); // first row is header, so ignore
		getline(lockdownfile, str); // get first row of data
		boost::split(fields, str, boost::is_any_of(","));
		if (!fields[0].empty() && !fields[47].empty()) {
			cumlockdowns[CountryIndex(code, fields[0])] = cumlockdowns[CountryIndex(code, fields[0])] + stod(fields[47]);
		}
		while (getline(lockdownfile, str)) {
			if (!str.empty()) {
				try {
					boost::split(fields, str, boost::is_any_of(","));
					if (!fields[0].empty() && !fields[47].empty()) {
						cumlockdowns[CountryIndex(code, fields[0])] = cumlockdowns[CountryIndex(code, fields[0])] + stod(fields[47]);
					}
				}
				catch (const std::exception& e) {
					std::cout << e.what();
				}
			}
		}
	}
	lockdownfile.clear();
	lockdownfile.seekg(0);
	std::cout << ".";






	///////////
	// MASKS //
	///////////
	std::vector<std::vector<double>> masks(numentities, std::vector<double>(numdates, nan("")));

	std::string masksfilename;
	masksfilename = "face-covering-policies-covid.csv";
	std::ifstream masksfile;
	masksfile.open(masksfilename);
	while (!masksfile.eof()) {
		getline(masksfile, str); // first row is header, so ignore
		while (getline(masksfile, str)) {
			if (!str.empty()) {
				try {
					boost::split(fields, str, boost::is_any_of(","));
					d = from_simple_string(fields[2]);
					dd = d - firstdate;
					dateint = dd.days();
					if (!fields[1].empty() && !fields[3].empty() && (dateint < numdates)) {
						masks[CountryIndex(code, fields[1])][dateint] = stod(fields[3]);
					}
				}
				catch (const std::exception& e) {
					std::cout << e.what();
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

	std::vector<double> cummasks(numentities);
	// Afghanistan,AFG,2020-01-01,0
	while (!masksfile.eof()) {
		getline(masksfile, str); // first row is header, so ignore
		getline(masksfile, str); // get first row of data
		boost::split(fields, str, boost::is_any_of(","));
		if (!fields[1].empty() && !fields[3].empty()) {
			cummasks[CountryIndex(code, fields[1])] = cummasks[CountryIndex(code, fields[1])] + stod(fields[3]);
		}
		while (getline(masksfile, str)) {
			if (!str.empty()) {
				try {
					boost::split(fields, str, boost::is_any_of(","));
					if (!fields[1].empty() && !fields[3].empty()) {
						cummasks[CountryIndex(code, fields[1])] = cummasks[CountryIndex(code, fields[1])] + stod(fields[3]);
					}
				}
				catch (const std::exception& e) {
					std::cout << e.what();
				}
			}
		}
	}
	masksfile.clear();
	masksfile.seekg(0);
	std::cout << ".";







	//////////////////
	// COVID DEATHS //
	//////////////////

	std::vector<double> cumcoviddeaths(numentities);
	// COVID-19 deaths
	// AFG,Asia,Afghanistan,2020-01-03,,0.0,,,0.0
	while (!coviddeathsfile.eof()) {
		getline(coviddeathsfile, str); // first row is header, so ignore
		while (getline(coviddeathsfile, str)) {
			if (!str.empty()) {
				try {
					boost::split(fields, str, boost::is_any_of(","));
					d = from_simple_string(fields[3]);
					dd = d - firstdate;
					dateint = dd.days();
					if (!fields[0].empty() && !fields[15].empty() && (dateint < numdates)) {
						coviddeaths[CountryIndex(code, fields[0])][dateint] = stod(fields[15]);
						cumcoviddeaths[CountryIndex(code, fields[0])] = cumcoviddeaths[CountryIndex(code, fields[0])] + stod(fields[15]);
					}
				}
				catch (const std::exception& e) {
					std::cout << e.what();
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
	std::sort(tempcoviddeaths.begin(), tempcoviddeaths.end());
	tcd1 = tempcoviddeaths[round((tempcoviddeaths.size() / 3) - 1)];
	tcd2 = tempcoviddeaths[round((2 * tempcoviddeaths.size() / 3) - 1)];
	for (unsigned int i = 0; i < numentities; i++)
		if (cumcoviddeaths[i] != 0) {
			if (cumcoviddeaths[i] <= tcd1)
				coviddeathslow.push_back(i);
			else if (cumcoviddeaths[i] <= tcd2)
				coviddeathsmed.push_back(i);
			else
				coviddeathshigh.push_back(i);
		}

	outputfile << "Countries with a low COVID-19 death rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million" << std::endl;
	for (unsigned int i = 0; i < coviddeathslow.size(); i++)
		display.push_back(entity[coviddeathslow[i]] + "\t" + tostring(cumcoviddeaths[coviddeathslow[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();

	outputfile << "Countries with a medium COVID-19 death rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million" << std::endl;
	for (unsigned int i = 0; i < coviddeathsmed.size(); i++)
		display.push_back(entity[coviddeathsmed[i]] + "\t" + tostring(cumcoviddeaths[coviddeathsmed[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();

	outputfile << "Countries with a high COVID-19 death rateh" << std::endl;
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

	std::vector<std::vector<double>> excessmortalitypscore(numentities, std::vector<double>(numdates, nan("")));

	std::vector<double> HRVexcessmortalitypscore_0_14(numdates, nan(""));
	std::vector<double> HRVexcessmortalitypscore_15_64(numdates, nan(""));
	std::vector<double> HRVexcessmortalitypscore_65_74(numdates, nan(""));
	std::vector<double> HRVexcessmortalitypscore_75_84(numdates, nan(""));
	std::vector<double> HRVexcessmortalitypscore_85p(numdates, nan(""));

	std::vector<double> HUNexcessmortalitypscore_0_14(numdates, nan(""));
	std::vector<double> HUNexcessmortalitypscore_15_64(numdates, nan(""));
	std::vector<double> HUNexcessmortalitypscore_65_74(numdates, nan(""));
	std::vector<double> HUNexcessmortalitypscore_75_84(numdates, nan(""));
	std::vector<double> HUNexcessmortalitypscore_85p(numdates, nan(""));

	std::vector<double> HRVexcessmortalitypscore_0_14interpolated(numdates, nan(""));
	std::vector<double> HRVexcessmortalitypscore_15_64interpolated(numdates, nan(""));
	std::vector<double> HRVexcessmortalitypscore_65_74interpolated(numdates, nan(""));
	std::vector<double> HRVexcessmortalitypscore_75_84interpolated(numdates, nan(""));
	std::vector<double> HRVexcessmortalitypscore_85pinterpolated(numdates, nan(""));

	std::vector<double> HUNexcessmortalitypscore_0_14interpolated(numdates, nan(""));
	std::vector<double> HUNexcessmortalitypscore_15_64interpolated(numdates, nan(""));
	std::vector<double> HUNexcessmortalitypscore_65_74interpolated(numdates, nan(""));
	std::vector<double> HUNexcessmortalitypscore_75_84interpolated(numdates, nan(""));
	std::vector<double> HUNexcessmortalitypscore_85pinterpolated(numdates, nan(""));

    // Entity,Code,Day,p_avg_0_14,p_avg_15_64,p_avg_65_74,p_avg_75_84,p_avg_85p,p_avg_all_ages
    // Croatia,HRV,2020-01-05,14.29,-7.84,-3.74,-28.37,-8.06,-14.47
	std::string excessmortalitypscorefilename;
	excessmortalitypscorefilename = "excess-mortality-p-scores-average-baseline-by-age.csv";
	std::ifstream excessmortalitypscorefile;
	excessmortalitypscorefile.open(excessmortalitypscorefilename);
	while (!excessmortalitypscorefile.eof()) {
		getline(excessmortalitypscorefile, str); // first row is header, so ignore
		getline(excessmortalitypscorefile, str); // get first row of data
		boost::split(fields, str, boost::is_any_of(","));
		d = from_simple_string(fields[2]);
		dd = d - firstdate;
		dateint = dd.days();
		if (!fields[1].empty() && !fields[8].empty() && (dateint < numdates)) {
			excessmortalitypscore[CountryIndex(code, fields[1])][dateint] = stod(fields[8]);

			if (fields[1] == "HRV") {
				HRVexcessmortalitypscore_0_14[dateint] = stod(fields[3]);
				HRVexcessmortalitypscore_15_64[dateint] = stod(fields[4]);
				HRVexcessmortalitypscore_65_74[dateint] = stod(fields[5]);
				HRVexcessmortalitypscore_75_84[dateint] = stod(fields[6]);
				HRVexcessmortalitypscore_85p[dateint] = stod(fields[7]);
			}


			if (fields[1] == "HUN") {
				HUNexcessmortalitypscore_0_14[dateint] = stod(fields[3]);
				HUNexcessmortalitypscore_15_64[dateint] = stod(fields[4]);
				HUNexcessmortalitypscore_65_74[dateint] = stod(fields[5]);
				HUNexcessmortalitypscore_75_84[dateint] = stod(fields[6]);
				HUNexcessmortalitypscore_85p[dateint] = stod(fields[7]);
			}

			cumexcess[CountryIndex(code, fields[1])] = cumexcess[CountryIndex(code, fields[1])] + stod(fields[8]);

			if (dateint >= 342)
				cumexcesspostvax[CountryIndex(code, fields[1])] = cumexcesspostvax[CountryIndex(code, fields[1])] + stod(fields[8]);

			if (dateint >= 366 && dateint <= 730)
				cumexcess2021[CountryIndex(code, fields[1])] = cumexcess2021[CountryIndex(code, fields[1])] + stod(fields[8]);

			if (dateint >= 731 && dateint <= 1095)
				cumexcess2022[CountryIndex(code, fields[1])] = cumexcess2022[CountryIndex(code, fields[1])] + stod(fields[8]);

			if (dateint >= 1096)
				cumexcess2023[CountryIndex(code, fields[1])] = cumexcess2023[CountryIndex(code, fields[1])] + stod(fields[8]);

			sumem[CountryIndex(code, fields[1])]++;
			if (dateint >= 342)
				sumempostvax[CountryIndex(code, fields[1])]++;
			if (dateint >= 366 && dateint <= 730)
				sumem2021[CountryIndex(code, fields[1])]++;
			if (dateint >= 731 && dateint <= 1095)
				sumem2022[CountryIndex(code, fields[1])]++;
			if (dateint >= 1096)
				sumem2023[CountryIndex(code, fields[1])]++;
		}




		while (getline(excessmortalitypscorefile, str)) {
			if (!str.empty()) {
				try {
					boost::split(fields, str, boost::is_any_of(","));
					d = from_simple_string(fields[2]);
					dd = d - firstdate;
					dateint = dd.days();
					if (!fields[1].empty() && !fields[8].empty() && !(std::find(code.begin(), code.end(), fields[1]) == code.end()) && (dateint < numdates)) {
						excessmortalitypscore[CountryIndex(code, fields[1])][dateint] = stod(fields[8]);

						if (fields[1] == "HRV") {
							HRVexcessmortalitypscore_0_14[dateint] = stod(fields[3]);
							HRVexcessmortalitypscore_15_64[dateint] = stod(fields[4]);
							HRVexcessmortalitypscore_65_74[dateint] = stod(fields[5]);
							HRVexcessmortalitypscore_75_84[dateint] = stod(fields[6]);
							HRVexcessmortalitypscore_85p[dateint] = stod(fields[7]);
						}

						if (fields[1] == "HUN") {
							HUNexcessmortalitypscore_0_14[dateint] = stod(fields[3]);
							HUNexcessmortalitypscore_15_64[dateint] = stod(fields[4]);
							HUNexcessmortalitypscore_65_74[dateint] = stod(fields[5]);
							HUNexcessmortalitypscore_75_84[dateint] = stod(fields[6]);
							HUNexcessmortalitypscore_85p[dateint] = stod(fields[7]);
						}

						cumexcess[CountryIndex(code, fields[1])] = cumexcess[CountryIndex(code, fields[1])] + stod(fields[8]);

						if (dateint >= 342)
							cumexcesspostvax[CountryIndex(code, fields[1])] = cumexcesspostvax[CountryIndex(code, fields[1])] + stod(fields[8]);

						if (dateint >= 366 && dateint <= 730)
							cumexcess2021[CountryIndex(code, fields[1])] = cumexcess2021[CountryIndex(code, fields[1])] + stod(fields[8]);

						if (dateint >= 731 && dateint <= 1095)
							cumexcess2022[CountryIndex(code, fields[1])] = cumexcess2022[CountryIndex(code, fields[1])] + stod(fields[8]);

						if (dateint >= 1096)
							cumexcess2023[CountryIndex(code, fields[1])] = cumexcess2023[CountryIndex(code, fields[1])] + stod(fields[8]);

						sumem[CountryIndex(code, fields[1])]++;
						if (dateint >= 342)
							sumempostvax[CountryIndex(code, fields[1])]++;
						if (dateint >= 366 && dateint <= 730)
							sumem2021[CountryIndex(code, fields[1])]++;
						if (dateint >= 731 && dateint <= 1095)
							sumem2022[CountryIndex(code, fields[1])]++;
						if (dateint >= 1096)
							sumem2023[CountryIndex(code, fields[1])]++;
					}
				}
				catch (const std::exception& e) {
					std::cout << e.what();
				}
			}
		}
	}
	std::cout << ".";
	for (int i = 0; i < cumexcess.size(); i++)
		meanexcess[i] = cumexcess[i] / sumem[i];

	for (int i = 0; i < cumexcesspostvax.size(); i++)
		meanexcesspostvax[i] = cumexcesspostvax[i] / sumempostvax[i];

	for (int i = 0; i < cumexcess2021.size(); i++)
		meanexcess2021[i] = cumexcess2021[i] / sumem2021[i];

	for (int i = 0; i < cumexcess2022.size(); i++)
		meanexcess2022[i] = cumexcess2022[i] / sumem2022[i];

	for (int i = 0; i < cumexcess2023.size(); i++)
		meanexcess2023[i] = cumexcess2023[i] / sumem2023[i];

	excessmortalitypscorefile.clear();
	excessmortalitypscorefile.seekg(0);



	std::vector<std::vector<double>> excessmortalitypscoreinterpolated(numentities, std::vector<double>(numdates, nan("")));



	excessmortalitypscoreinterpolated = excessmortalitypscore;
	std::vector<double> x;
	std::vector<double> y;
	for (unsigned int c = 0; c < excessmortalitypscore.size(); c++) {
		for (unsigned int d = 0; d < excessmortalitypscore[c].size(); d++) {
			if (!isnan(excessmortalitypscore[c][d])) {
				x.push_back(double(d));
				y.push_back(excessmortalitypscore[c][d]);
			}
		}
		int first = -1;
		int last = -1;
		bool firstcase = true;
		for (unsigned int d = 0; d < excessmortalitypscoreinterpolated[c].size(); d++)
			if (!isnan(excessmortalitypscore[c][d])) {
				if (firstcase == true) {
					first = d;
					firstcase = false;
				}
				last = d;
			}
		if (x.size() > 0) {
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
	std::ofstream coviddeathslowcdfile;
	coviddeathslowcdfile.open(coviddeathslowcdfilename);

	std::string coviddeathslowemfilename;
	coviddeathslowemfilename = "coviddeathslowem.txt";
	std::ofstream coviddeathslowemfile;
	coviddeathslowemfile.open(coviddeathslowemfilename);
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
	std::ofstream coviddeathsmedcdfile;
	coviddeathsmedcdfile.open(coviddeathsmedcdfilename);
	std::string coviddeathsmedemfilename;
	coviddeathsmedemfilename = "coviddeathsmedem.txt";
	std::ofstream coviddeathsmedemfile;
	coviddeathsmedemfile.open(coviddeathsmedemfilename);
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
	std::ofstream coviddeathshighcdfile;
	coviddeathshighcdfile.open(coviddeathshighcdfilename);
	std::string coviddeathshighemfilename;
	coviddeathshighemfilename = "coviddeathshighem.txt";
	std::ofstream coviddeathshighemfile;
	coviddeathshighemfile.open(coviddeathshighemfilename);
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
	std::vector<double> cumvax(numentities);
	std::vector<double> cumvax2020(numentities);
	std::vector<double> cumvax20202021(numentities);
	std::vector<double> cumvax20202022(numentities);
	// covid-vaccination-doses-per-capita.csv
	// Afghanistan,AFG,2021-02-22,0
	std::string cumvaxfilename;
	cumvaxfilename = "covid-vaccination-doses-per-capita.csv";
	std::ifstream cumvaxfile;
	cumvaxfile.open(cumvaxfilename);
	while (!cumvaxfile.eof()) {
		getline(cumvaxfile, str); // first row is header, so ignore
		getline(cumvaxfile, str); // get first row of data
		boost::split(fields, str, boost::is_any_of(","));
		d = from_simple_string(fields[2]);
		dd = d - firstdate;
		dateint = dd.days();
		if (!fields[1].empty() && !fields[3].empty()) {
			cumvax[CountryIndex(code, fields[1])] = stod(fields[3]);
			if (dateint <= 730)
				cumvax20202021[CountryIndex(code, fields[1])] = stod(fields[3]);
			if (dateint <= 1095)
				cumvax20202022[CountryIndex(code, fields[1])] = stod(fields[3]);
		}
		while (getline(cumvaxfile, str)) {
			if (!str.empty()) {
				try {
					boost::split(fields, str, boost::is_any_of(","));
					d = from_simple_string(fields[2]);
					dd = d - firstdate;
					dateint = dd.days();
					if (!fields[1].empty() && !fields[3].empty()) {
						cumvax[CountryIndex(code, fields[1])] = stod(fields[3]);
						if (dateint <= 730)
							cumvax20202021[CountryIndex(code, fields[1])] = stod(fields[3]);
						if (dateint <= 1095)
							cumvax20202022[CountryIndex(code, fields[1])] = stod(fields[3]);
					}
				}
				catch (const std::exception& e) {
					std::cout << e.what();
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
	outputfile << "Continent" << "\t" << "Entity" << "\t" << "Code" << "\t" << "Cumulative lockdown stringency index" << "\t" << "Cumulative face covering policies index" << "\t" << "Total vaccinations per hundred" << "\t" << "Total COVID-19 deaths per million" << "\t" << "Mean excess mortality P-scores" << std::endl;
	for (unsigned int i = 0; i < numentities; i++) {
		if (!isnan(meanexcess[i]))
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
	outputfile << "Continent" << "\t" << "Entity" << "\t" << "Code" << "\t" << "Vaccinations per hundred in 2020" << "\t" << "Vaccinations per hundred in 2021" << "\t" << "Vaccinations per hundred in 2022" << "\t" << "Total vaccinations per hundred" << "\t" << "Mean post vaccination excess mortality P-scores" << "\t" << "Mean 2021 excess mortality P-scores" << "\t" << "Mean 2022 excess mortality P-scores" << "\t" << "Mean 2023 excess mortality P-scores" << "\t" << "Mean excess mortality P-scores" << std::endl;
	for (unsigned int i = 0; i < numentities; i++) {

		if (!isnan(meanexcesspostvax[i]))
			excesspostvaxstring = tostring(meanexcesspostvax[i]);
		else
			excesspostvaxstring = "nan";

		if (!isnan(meanexcess2021[i]))
			excess2021string = tostring(meanexcess2021[i]);
		else
			excess2021string = "nan";

		if (!isnan(meanexcess2022[i]))
			excess2022string = tostring(meanexcess2022[i]);
		else
			excess2022string = "nan";

		if (!isnan(meanexcess2023[i]))
			excess2023string = tostring(meanexcess2023[i]);
		else
			excess2023string = "nan";

		if (!isnan(meanexcess[i]))
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


	std::string vaxfilename;
	vaxfilename = "daily-covid-vaccination-doses-per-capita.csv";
	std::ifstream vaxfile;
	vaxfile.open(vaxfilename);

	// Afghanistan,AFG,2021-02-22,0
	while (!vaxfile.eof()) {
		getline(vaxfile, str); // first row is header, so ignore
		while (getline(vaxfile, str)) {
			if (!str.empty()) {
				try {
					boost::split(fields, str, boost::is_any_of(","));
					d = from_simple_string(fields[2]);
					dd = d - firstdate;
					dateint = dd.days();
					if (!fields[1].empty() && !fields[3].empty() && (dateint < numdates)) {
						vaccinations[CountryIndex(code, fields[1])][dateint] = stod(fields[3]);
					}
				}
				catch (const std::exception& e) {
					std::cout << e.what();
				}
			}
		}
	}

	vaxfile.clear();
	vaxfile.seekg(0);
	std::cout << ".";




	std::vector<double> tempvax;

	outputfile << "Countries for clustering:" << std::endl;
	for (unsigned int i = 0; i < numentities; i++)
		if (cumvax[i] > 0 && !continent[i].empty()) { // ignore all zero vax (bad data) and non-countries
			outputfile << entity[i] << '\t';
			tempvax.push_back(cumvax[i]);
		}
	outputfile << std::endl;
	outputfile << std::endl;

	std::sort(tempvax.begin(), tempvax.end());
	tv1 = tempvax[round((tempvax.size() / 3) - 1)];
	tv2 = tempvax[round((2 * tempvax.size() / 3) - 1)];




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







	outputfile << "Countries with a low vaccination rate" << std::endl;
	outputfile << "Country\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < vaxlow.size(); i++)
		display.push_back(entity[vaxlow[i]] + "\t" + tostring(cumvax[vaxlow[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();


	outputfile << "Countries with a medium Vaccination rate" << std::endl;
	outputfile << "Country\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < vaxmed.size(); i++)
		display.push_back(entity[vaxmed[i]] + "\t" + tostring(cumvax[vaxmed[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();

	outputfile << "Countries with a high vaccination rate" << std::endl;
	outputfile << "Country\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < vaxhigh.size(); i++)
		display.push_back(entity[vaxhigh[i]] + "\t" + tostring(cumvax[vaxhigh[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();



	std::string vaxlowvxfilename;
	vaxlowvxfilename = "vaxlowvx.txt";
	std::ofstream vaxlowvxfile;
	vaxlowvxfile.open(vaxlowvxfilename);

	std::string vaxlowcdfilename;
	vaxlowcdfilename = "vaxlowcd.txt";
	std::ofstream vaxlowcdfile;
	vaxlowcdfile.open(vaxlowcdfilename);

	std::string vaxlowemfilename;
	vaxlowemfilename = "vaxlowem.txt";
	std::ofstream vaxlowemfile;
	vaxlowemfile.open(vaxlowemfilename);



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
	std::ofstream vaxmedvxfile;
	vaxmedvxfile.open(vaxmedvxfilename);

	std::string vaxmedcdfilename;
	vaxmedcdfilename = "vaxmedcd.txt";
	std::ofstream vaxmedcdfile;
	vaxmedcdfile.open(vaxmedcdfilename);

	std::string vaxmedemfilename;
	vaxmedemfilename = "vaxmedem.txt";
	std::ofstream vaxmedemfile;
	vaxmedemfile.open(vaxmedemfilename);

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
	std::ofstream vaxhighvxfile;
	vaxhighvxfile.open(vaxhighvxfilename);

	std::string vaxhighcdfilename;
	vaxhighcdfilename = "vaxhighcd.txt";
	std::ofstream vaxhighcdfile;
	vaxhighcdfile.open(vaxhighcdfilename);

	std::string vaxhighemfilename;
	vaxhighemfilename = "vaxhighem.txt";
	std::ofstream vaxhighemfile;
	vaxhighemfile.open(vaxhighemfilename);

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
	std::ofstream clusteringcdfile;
	clusteringcdfile.open(clusteringcdfilename);
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
	std::ofstream clusteringemfile;
	clusteringemfile.open(clusteringemfilename);
	for (unsigned int i = 0; i < numentities; i++)
		if (cumvax[i] > 0 && !continent[i].empty()) {
			for (int d = 0; d < numprevaxdates; d++)
				clusteringemfile << excessmortalitypscoreinterpolated[i][d] << "\t";
	        clusteringemfile << "\n";
		}
	clusteringemfile.flush();
	clusteringemfile.close();

	std::cout << std::endl;
	std::cout << "1) Copy this program's output files to the directory containing clustering.py." << std::endl;
	std::cout << "2) Run clustering.py." << std::endl;
	std::cout << "3) Copy clusters.txt from the clustering.py directory to the directory containg OurWorldinData.exe." << std::endl;
	std::cout << "4) ";
	system("pause");
	


	int mc = 19;
	int c;

	// std::vector<std::vector<double>> coviddeaths(numentities, std::vector<double>(numdates, nan("")));
	std::vector<std::vector<int>> clustcd(mc);
	std::vector<std::vector<int>> clustem(mc);

	std::string clustersfilename;
	clustersfilename = "clusters.txt";
	std::ifstream clustersfile;
	clustersfile.open(clustersfilename);
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
		if (cumlockdowns[i] != 0 && !continent[i].empty()) // ignore all zero values and non countries
			templockdown.push_back(cumlockdowns[i]);
	std::sort(templockdown.begin(), templockdown.end());
	tl1 = templockdown[round((templockdown.size() / 3) - 1)];
	tl2 = templockdown[round((2 * templockdown.size() / 3) - 1)];
	for (unsigned int i = 0; i < numentities; i++)
		if (cumlockdowns[i] != 0) {
			if (cumlockdowns[i] <= tl1)
				lockdownlow.push_back(i);
			else if (cumlockdowns[i] <= tl2)
				lockdownmed.push_back(i);
			else
				lockdownhigh.push_back(i);
		}


	outputfile << "Countries with low lockdown stringency" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency" << std::endl;
	for (unsigned int i = 0; i < lockdownlow.size(); i++)
		display.push_back(entity[lockdownlow[i]] + "\t" + tostring(cumlockdowns[lockdownlow[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();

	outputfile << "Countries with medium lockdown stringency" << std::endl;
	outputfile << "Countr // reads in 19 linesn stringency" << std::endl;
	for (unsigned int i = 0; i < lockdownmed.size(); i++)
		display.push_back(entity[lockdownmed[i]] + "\t" + tostring(cumlockdowns[lockdownmed[i]]));
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();

	outputfile << "Countries with high lockdown stringency" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency" << std::endl;
	for (unsigned int i = 0; i < lockdownhigh.size(); i++)
		display.push_back(entity[lockdownhigh[i]] + "\t" + tostring(cumlockdowns[lockdownhigh[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();

	std::string lockdownlowldfilename;
	lockdownlowldfilename = "lockdownlowld.txt";
	std::ofstream lockdownlowldfile;
	lockdownlowldfile.open(lockdownlowldfilename);
	std::string lockdownlowemfilename;
	lockdownlowemfilename = "lockdownlowem.txt";
	std::ofstream lockdownlowemfile;
	lockdownlowemfile.open(lockdownlowemfilename);
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
	std::ofstream lockdownmedldfile;
	lockdownmedldfile.open(lockdownmedldfilename);
	std::string lockdownmedemfilename;
	lockdownmedemfilename = "lockdownmedem.txt";
	std::ofstream lockdownmedemfile;
	lockdownmedemfile.open(lockdownmedemfilename);
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
	std::ofstream lockdownhighldfile;
	lockdownhighldfile.open(lockdownhighldfilename);
	std::string lockdownhighemfilename;
	lockdownhighemfilename = "lockdownhighem.txt";
	std::ofstream lockdownhighemfile;
	lockdownhighemfile.open(lockdownhighemfilename);
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
	std::sort(tempmasks.begin(), tempmasks.end());
	tm1 = tempmasks[round((tempmasks.size() / 3) - 1)];
	tm2 = tempmasks[round((2 * tempmasks.size() / 3) - 1)];
	for (unsigned int i = 0; i < numentities; i++)
		if (cummasks[i] != 0) {
			if (cummasks[i] <= tm1)
				maskslow.push_back(i);
			else if (cummasks[i] <= tm2)
				masksmed.push_back(i);
			else
				maskshigh.push_back(i);
		}
	std::cout << ".";
	outputfile << "Countries with low mask policies" << std::endl;
	outputfile << "Country\tCumulative facial coverings index" << std::endl;
	for (unsigned int i = 0; i < maskslow.size(); i++)
		display.push_back(entity[maskslow[i]] + "\t" + tostring(cummasks[maskslow[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();


	outputfile << "Countries with medium mask policies" << std::endl;
	outputfile << "Country\tCumulative facial coverings index" << std::endl;
	for (unsigned int i = 0; i < masksmed.size(); i++)
		display.push_back(entity[masksmed[i]] + "\t" + tostring(cummasks[masksmed[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();

	outputfile << "Countries with high mask policies" << std::endl;
	outputfile << "Country\tCumulative facial coverings index" << std::endl;
	for (unsigned int i = 0; i < maskshigh.size(); i++)
		display.push_back(entity[maskshigh[i]] + "\t" + tostring(cummasks[maskshigh[i]]));
	std::sort(display.begin(), display.end());
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	display.clear();


	std::string maskslowmafilename;
	maskslowmafilename = "maskslowma.txt";
	std::ofstream maskslowmafile;
	maskslowmafile.open(maskslowmafilename);
	std::string maskslowemfilename;
	maskslowemfilename = "maskslowem.txt";
	std::ofstream maskslowemfile;
	maskslowemfile.open(maskslowemfilename);
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
	std::ofstream masksmedmafile;
	masksmedmafile.open(masksmedmafilename);
	std::string masksmedemfilename;
	masksmedemfilename = "masksmedem.txt";
	std::ofstream masksmedemfile;
	masksmedemfile.open(masksmedemfilename);
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
	std::ofstream maskshighmafile;
	maskshighmafile.open(maskshighmafilename);
	std::string maskshighemfilename;
	maskshighemfilename = "maskshighem.txt";
	std::ofstream maskshighemfile;
	maskshighemfile.open(maskshighemfilename);
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
	outputfile << "Countries with low face covering policy stringency and a low vaccination rate" << std::endl;
	outputfile << "Country\tCumulative face covering policy stringency\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string maskslowvaxlowfilename;
	maskslowvaxlowfilename = "maskslowvaxlow.txt";
	std::ofstream maskslowvaxlowfile;
	maskslowvaxlowfile.open(maskslowvaxlowfilename);
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
	outputfile << "Countries with a low face covering policy stringency and a medium vaccination rate" << std::endl;
	outputfile << "Country\tCumulative face covering policy index\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string maskslowvaxmedfilename;
	maskslowvaxmedfilename = "maskslowvaxmed.txt";
	std::ofstream maskslowvaxmedfile;
	maskslowvaxmedfile.open(maskslowvaxmedfilename);
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
	outputfile << "Countries with a low face covering policy stringency and a high vaccination rate" << std::endl;
	outputfile << "Country\tCumulative face covering policy index\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string maskslowvaxhighfilename;
	maskslowvaxhighfilename = "maskslowvaxhigh.txt";
	std::ofstream maskslowvaxhighfile;
	maskslowvaxhighfile.open(maskslowvaxhighfilename);
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
	outputfile << "Countries with a medium face covering policy stringency and a low vaccination rate" << std::endl;
	outputfile << "Country\tCumulative face covering policy index\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string masksmedvaxlowfilename;
	masksmedvaxlowfilename = "masksmedvaxlow.txt";
	std::ofstream masksmedvaxlowfile;
	masksmedvaxlowfile.open(masksmedvaxlowfilename);
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
	outputfile << "Countries with a medium face covering policy stringency and a medium vaccination rate" << std::endl;
	outputfile << "Country\tCumulative face covering policy index\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string masksmedvaxmedfilename;
	masksmedvaxmedfilename = "masksmedvaxmed.txt";
	std::ofstream masksmedvaxmedfile;
	masksmedvaxmedfile.open(masksmedvaxmedfilename);
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
	outputfile << "Countries with a medium face covering policy stringency and a high vaccination rate" << std::endl;
	outputfile << "Country\tCumulative face covering policy index\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string masksmedvaxhighfilename;
	masksmedvaxhighfilename = "masksmedvaxhigh.txt";
	std::ofstream masksmedvaxhighfile;
	masksmedvaxhighfile.open(masksmedvaxhighfilename);
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
	outputfile << "Countries with a high face covering policy stringency and a low vaccination rate" << std::endl;
	outputfile << "Country\tCumulative face covering policy index\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string maskshighvaxlowfilename;
	maskshighvaxlowfilename = "maskshighvaxlow.txt";
	std::ofstream maskshighvaxlowfile;
	maskshighvaxlowfile.open(maskshighvaxlowfilename);
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
	outputfile << "Countries with a high face covering policy stringency and a medium vaccination rate" << std::endl;
	outputfile << "Country\tCumulative face covering policy index\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string maskshighvaxmedfilename;
	maskshighvaxmedfilename = "maskshighvaxmed.txt";
	std::ofstream maskshighvaxmedfile;
	maskshighvaxmedfile.open(maskshighvaxmedfilename);
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
	outputfile << "Countries with a high face covering policy stringency and a high vaccination rate" << std::endl;
	outputfile << "Country\tCumulative face covering policy index\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string maskshighvaxhighfilename;
	maskshighvaxhighfilename = "maskshighvaxhigh.txt";
	std::ofstream maskshighvaxhighfile;
	maskshighvaxhighfile.open(maskshighvaxhighfilename);
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
	outputfile << "Countries with a low lockdown stringency and a low vaccination rate" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownlowvaxlowfilename;
	lockdownlowvaxlowfilename = "lockdownlowvaxlow.txt";
	std::ofstream lockdownlowvaxlowfile;
	lockdownlowvaxlowfile.open(lockdownlowvaxlowfilename);
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
	outputfile << "Countries with a low lockdown stringency and a medium vaccination rate" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownlowvaxmedfilename;
	lockdownlowvaxmedfilename = "lockdownlowvaxmed.txt";
	std::ofstream lockdownlowvaxmedfile;
	lockdownlowvaxmedfile.open(lockdownlowvaxmedfilename);
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
	outputfile << "Countries with a low lockdown stringency and a high vaccination rate" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownlowvaxhighfilename;
	lockdownlowvaxhighfilename = "lockdownlowvaxhigh.txt";
	std::ofstream lockdownlowvaxhighfile;
	lockdownlowvaxhighfile.open(lockdownlowvaxhighfilename);
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
	outputfile << "Countries with a medium lockdown stringency and a low vaccination rate" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownmedvaxlowfilename;
	lockdownmedvaxlowfilename = "lockdownmedvaxlow.txt";
	std::ofstream lockdownmedvaxlowfile;
	lockdownmedvaxlowfile.open(lockdownmedvaxlowfilename);
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
	outputfile << "Countries with a medium lockdown stringency and a medium vaccination rate" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownmedvaxmedfilename;
	lockdownmedvaxmedfilename = "lockdownmedvaxmed.txt";
	std::ofstream lockdownmedvaxmedfile;
	lockdownmedvaxmedfile.open(lockdownmedvaxmedfilename);
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
	outputfile << "Countries with a medium lockdown stringency and a high vaccination rate" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownmedvaxhighfilename;
	lockdownmedvaxhighfilename = "lockdownmedvaxhigh.txt";
	std::ofstream lockdownmedvaxhighfile;
	lockdownmedvaxhighfile.open(lockdownmedvaxhighfilename);
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
	outputfile << "Countries with a high lockdown stringency and a low vaccination rate" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownhighvaxlowfilename;
	lockdownhighvaxlowfilename = "lockdownhighvaxlow.txt";
	std::ofstream lockdownhighvaxlowfile;
	lockdownhighvaxlowfile.open(lockdownhighvaxlowfilename);
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
	outputfile << "Countries with a high lockdown stringency and a medium vaccination rate" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownhighvaxmedfilename;
	lockdownhighvaxmedfilename = "lockdownhighvaxmed.txt";
	std::ofstream lockdownhighvaxmedfile;
	lockdownhighvaxmedfile.open(lockdownhighvaxmedfilename);
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
	outputfile << "Countries with a high lockdown stringency and a high vaccination rate" << std::endl;
	outputfile << "Country\tCumulative lockdown stringency\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string lockdownhighvaxhighfilename;
	lockdownhighvaxhighfilename = "lockdownhighvaxhigh.txt";
	std::ofstream lockdownhighvaxhighfile;
	lockdownhighvaxhighfile.open(lockdownhighvaxhighfilename);
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
	outputfile << "Countries with a low COVID-19 death rate and a low vaccination rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathslowvaxlowfilename;
	coviddeathslowvaxlowfilename = "coviddeathslowvaxlow.txt";
	std::ofstream coviddeathslowvaxlowfile;
	coviddeathslowvaxlowfile.open(coviddeathslowvaxlowfilename);
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
	outputfile << "Countries with a low COVID-19 death rate and a medium vaccination rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathslowvaxmedfilename;
	coviddeathslowvaxmedfilename = "coviddeathslowvaxmed.txt";
	std::ofstream coviddeathslowvaxmedfile;
	coviddeathslowvaxmedfile.open(coviddeathslowvaxmedfilename);
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
	outputfile << "Countries with a low COVID-19 death rate and a high vaccination rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathslowvaxhighfilename;
	coviddeathslowvaxhighfilename = "coviddeathslowvaxhigh.txt";
	std::ofstream coviddeathslowvaxhighfile;
	coviddeathslowvaxhighfile.open(coviddeathslowvaxhighfilename);
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
	outputfile << "Countries with a medium COVID-19 death rate and a low vaccination rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathsmedvaxlowfilename;
	coviddeathsmedvaxlowfilename = "coviddeathsmedvaxlow.txt";
	std::ofstream coviddeathsmedvaxlowfile;
	coviddeathsmedvaxlowfile.open(coviddeathsmedvaxlowfilename);
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
	outputfile << "Countries with a medium COVID-19 death rate and a medium vaccination rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathsmedvaxmedfilename;
	coviddeathsmedvaxmedfilename = "coviddeathsmedvaxmed.txt";
	std::ofstream coviddeathsmedvaxmedfile;
	coviddeathsmedvaxmedfile.open(coviddeathsmedvaxmedfilename);
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
	outputfile << "Countries with a medium COVID-19 death rate and a high vaccination rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathsmedvaxhighfilename;
	coviddeathsmedvaxhighfilename = "coviddeathsmedvaxhigh.txt";
	std::ofstream coviddeathsmedvaxhighfile;
	coviddeathsmedvaxhighfile.open(coviddeathsmedvaxhighfilename);
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
	outputfile << "Countries with a high COVID-19 death rate and a low vaccination rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathshighvaxlowfilename;
	coviddeathshighvaxlowfilename = "coviddeathshighvaxlow.txt";
	std::ofstream coviddeathshighvaxlowfile;
	coviddeathshighvaxlowfile.open(coviddeathshighvaxlowfilename);
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
	outputfile << "Countries with a high COVID-19 death rate and a medium vaccination rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathshighvaxmedfilename;
	coviddeathshighvaxmedfilename = "coviddeathshighvaxmed.txt";
	std::ofstream coviddeathshighvaxmedfile;
	coviddeathshighvaxmedfile.open(coviddeathshighvaxmedfilename);
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
	outputfile << "Countries with a high COVID-19 death rate and a high vaccination rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tTotal vaccinations per hundred" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathshighvaxhighfilename;
	coviddeathshighvaxhighfilename = "coviddeathshighvaxhigh.txt";
	std::ofstream coviddeathshighvaxhighfile;
	coviddeathshighvaxhighfile.open(coviddeathshighvaxhighfilename);
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
		display.push_back(entity[coviddeathslowlockdownlowvaxlow[i]] + "\t" + tostring(cumcoviddeaths[lockdownlowvaxlow[i]]) + "\t" + tostring(cumlockdowns[lockdownlowvaxlow[i]]) + "\t" + tostring(cumvax[lockdownlowvaxlow[i]]));
	std::sort(display.begin(), display.end());
	outputfile << "Countries with a low COVID-19 death rate, low lockdown stringency and low vaccination rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tCumulative lockdown stringency\tTotal vaccinations per 100" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathslowlockdownlowvaxlowfilename;
	coviddeathslowlockdownlowvaxlowfilename = "coviddeathslowlockdownlowvaxlow.txt";
	std::ofstream coviddeathslowlockdownlowvaxlowfile;
	coviddeathslowlockdownlowvaxlowfile.open(coviddeathslowlockdownlowvaxlowfilename);
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
	outputfile << "Countries with a high COVID-19 death rate, low lockdown stringency and low vaccination rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tCumulative lockdown stringency\tTotal vaccinations per 100" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathshighlockdownlowvaxlowfilename;
	coviddeathshighlockdownlowvaxlowfilename = "coviddeathshighlockdownlowvaxlow.txt";
	std::ofstream coviddeathshighlockdownlowvaxlowfile;
	coviddeathshighlockdownlowvaxlowfile.open(coviddeathshighlockdownlowvaxlowfilename);
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
	outputfile << "Countries with a low COVID-19 death rate, high lockdown stringency and low vaccination rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tCumulative lockdown stringency\tTotal vaccinations per 100" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathslowlockdownhighvaxlowfilename;
	coviddeathslowlockdownhighvaxlowfilename = "coviddeathslowlockdownhighvaxlow.txt";
	std::ofstream coviddeathslowlockdownhighvaxlowfile;
	coviddeathslowlockdownhighvaxlowfile.open(coviddeathslowlockdownhighvaxlowfilename);
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
	outputfile << "Countries with a low COVID-19 death rate, low lockdown stringency and high vaccination rate" << std::endl;
	outputfile << "Country\tTotal COVID-19 deaths per million\tCumulative lockdown stringency\tTotal vaccinations per 100" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;
	outputfile << std::endl;
	std::string coviddeathslowlockdownlowvaxhighfilename;
	coviddeathslowlockdownlowvaxhighfilename = "coviddeathslowlockdownlowvaxhigh.txt";
	std::ofstream coviddeathslowlockdownlowvaxhighfile;
	coviddeathslowlockdownlowvaxhighfile.open(coviddeathslowlockdownlowvaxhighfilename);
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
	std::ofstream countrylistfile;
	countrylistfile.open(countrylistfilename);

	std::string countryfilename;
	std::ofstream countryfile;
	for (int c = 0; c < entity.size(); c++) {
		country = entity[c];
		countrylistfile << country << std::endl;
		std::replace(country.begin(), country.end(), ' ', '_');
		countryfilename = country + ".txt";
		countryfile.open(countryfilename);
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





	// COUNTRY PAIRS

	int ci1, ci2;

	// Algeria vs Egypt CD
	ci1 = CountryIndex(code, "DZA");
	ci2 = CountryIndex(code, "EGY");
	std::string AlgeriaEgyptCDfilename;
	AlgeriaEgyptCDfilename = "AlgeriaEgyptCD.txt";
	std::ofstream AlgeriaEgyptCDfile;
	AlgeriaEgyptCDfile.open(AlgeriaEgyptCDfilename);
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
		AlgeriaEgyptCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Algeria vs Egypt EM
	ci1 = CountryIndex(code, "DZA");
	ci2 = CountryIndex(code, "EGY");
	std::string AlgeriaEgyptfilename;
	AlgeriaEgyptfilename = "AlgeriaEgyptEM.txt";
	std::ofstream AlgeriaEgyptfile;
	AlgeriaEgyptfile.open(AlgeriaEgyptfilename);
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
		AlgeriaEgyptfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Antigua and Barbuda vs Cuba EM
	ci1 = CountryIndex(code, "ATG");
	ci2 = CountryIndex(code, "CUB");
	std::string AntiguaandBarbudaCubafilename;
	AntiguaandBarbudaCubafilename = "AntiguaandBarbudaCubaEM.txt";
	std::ofstream AntiguaandBarbudaCubafile;
	AntiguaandBarbudaCubafile.open(AntiguaandBarbudaCubafilename);
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
		AntiguaandBarbudaCubafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Argentina vs Paraguay EM
	ci1 = CountryIndex(code, "ARG");
	ci2 = CountryIndex(code, "PRY");
	std::string ArgentinaParaguayfilename;
	ArgentinaParaguayfilename = "ArgentinaParaguayEM.txt";
	std::ofstream ArgentinaParaguayfile;
	ArgentinaParaguayfile.open(ArgentinaParaguayfilename);
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
		ArgentinaParaguayfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Armenia vs Azerbaijan CD
	ci1 = CountryIndex(code, "ARM");
	ci2 = CountryIndex(code, "AZE");
	std::string ArmeniaAzerbaijanCDfilename;
	ArmeniaAzerbaijanCDfilename = "ArmeniaAzerbaijanCD.txt";
	std::ofstream ArmeniaAzerbaijanCDfile;
	ArmeniaAzerbaijanCDfile.open(ArmeniaAzerbaijanCDfilename);
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
		ArmeniaAzerbaijanCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Armenia vs Azerbaijan EM
	ci1 = CountryIndex(code, "ARM");
	ci2 = CountryIndex(code, "AZE");
	std::string ArmeniaAzerbaijanfilename;
	ArmeniaAzerbaijanfilename = "ArmeniaAzerbaijanEM.txt";
	std::ofstream ArmeniaAzerbaijanfile;
	ArmeniaAzerbaijanfile.open(ArmeniaAzerbaijanfilename);
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
		ArmeniaAzerbaijanfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Australia vs French Polynesia EM
	ci1 = CountryIndex(code, "AUS");
	ci2 = CountryIndex(code, "PYF");
	std::string AustraliaFrenchPolynesiafilename;
	AustraliaFrenchPolynesiafilename = "AustraliaFrenchPolynesiaEM.txt";
	std::ofstream AustraliaFrenchPolynesiafile;
	AustraliaFrenchPolynesiafile.open(AustraliaFrenchPolynesiafilename);
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
		AustraliaFrenchPolynesiafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Australia vs New Caledonia EM
	ci1 = CountryIndex(code, "AUS");
	ci2 = CountryIndex(code, "NCL");
	std::string AustraliaNewCaledoniafilename;
	AustraliaNewCaledoniafilename = "AustraliaNewCaledoniaEM.txt";
	std::ofstream AustraliaNewCaledoniafile;
	AustraliaNewCaledoniafile.open(AustraliaNewCaledoniafilename);
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
		AustraliaNewCaledoniafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Australia vs New Zealand CD
	ci1 = CountryIndex(code, "AUS");
	ci2 = CountryIndex(code, "NZL");
	std::string AustraliaNewZealandCDfilename;
	AustraliaNewZealandCDfilename = "AustraliaNewZealandCD.txt";
	std::ofstream AustraliaNewZealandCDfile;
	AustraliaNewZealandCDfile.open(AustraliaNewZealandCDfilename);
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
		AustraliaNewZealandCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Australia vs New Zealand EM
	ci1 = CountryIndex(code, "AUS");
	ci2 = CountryIndex(code, "NZL");
	std::string AustraliaNewZealandfilename;
	AustraliaNewZealandfilename = "AustraliaNewZealandEM.txt";
	std::ofstream AustraliaNewZealandfile;
	AustraliaNewZealandfile.open(AustraliaNewZealandfilename);
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
		AustraliaNewZealandfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Australia vs Malaysia EM
	ci1 = CountryIndex(code, "AUS");
	ci2 = CountryIndex(code, "MYS");
	std::string AustraliaMalaysiafilename;
	AustraliaMalaysiafilename = "AustraliaMalaysiaEM.txt";
	std::ofstream AustraliaMalaysiafile;
	AustraliaMalaysiafile.open(AustraliaMalaysiafilename);
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
		AustraliaMalaysiafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Belarus vs Denmark CD
	ci1 = CountryIndex(code, "BLR");
	ci2 = CountryIndex(code, "DNK");
	std::string BelarusDenmarkCDfilename;
	BelarusDenmarkCDfilename = "BelarusDenmarkCD.txt";
	std::ofstream BelarusDenmarkCDfile;
	BelarusDenmarkCDfile.open(BelarusDenmarkCDfilename);
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
		BelarusDenmarkCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Belarus vs Denmark EM
	ci1 = CountryIndex(code, "BLR");
	ci2 = CountryIndex(code, "DNK");
	std::string BelarusDenmarkfilename;
	BelarusDenmarkfilename = "BelarusDenmarkEM.txt";
	std::ofstream BelarusDenmarkfile;
	BelarusDenmarkfile.open(BelarusDenmarkfilename);
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
		BelarusDenmarkfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Bhutan vs Singapore CD
	ci1 = CountryIndex(code, "BTN");
	ci2 = CountryIndex(code, "SGP");
	std::string BhutanSingaporeCDfilename;
	BhutanSingaporeCDfilename = "BhutanSingaporeCD.txt";
	std::ofstream BhutanSingaporeCDfile;
	BhutanSingaporeCDfile.open(BhutanSingaporeCDfilename);
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
		BhutanSingaporeCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Bhutan vs Singapore EM
	ci1 = CountryIndex(code, "BTN");
	ci2 = CountryIndex(code, "SGP");
	std::string BhutanSingaporefilename;
	BhutanSingaporefilename = "BhutanSingaporeEM.txt";
	std::ofstream BhutanSingaporefile;
	BhutanSingaporefile.open(BhutanSingaporefilename);
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
		BhutanSingaporefile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Bhutan vs Thailand CD
	ci1 = CountryIndex(code, "BTN");
	ci2 = CountryIndex(code, "THA");
	std::string BhutanThailandCDfilename;
	BhutanThailandCDfilename = "BhutanThailandCD.txt";
	std::ofstream BhutanThailandCDfile;
	BhutanThailandCDfile.open(BhutanThailandCDfilename);
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
		BhutanThailandCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Bhutan vs Thailand EM
	ci1 = CountryIndex(code, "BTN");
	ci2 = CountryIndex(code, "THA");
	std::string BhutanThailandfilename;
	BhutanThailandfilename = "BhutanThailandEM.txt";
	std::ofstream BhutanThailandfile;
	BhutanThailandfile.open(BhutanThailandfilename);
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
		BhutanThailandfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Bhutan vs Vietnam CD
	ci1 = CountryIndex(code, "BTN");
	ci2 = CountryIndex(code, "VNM");
	std::string BhutanVietnamCDfilename;
	BhutanVietnamCDfilename = "BhutanVietnamCD.txt";
	std::ofstream BhutanVietnamCDfile;
	BhutanVietnamCDfile.open(BhutanVietnamCDfilename);
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
		BhutanVietnamCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Burundi vs Eritrea CD
	ci1 = CountryIndex(code, "BDI");
	ci2 = CountryIndex(code, "ERI");
	std::string BurundiEritreaCDfilename;
	BurundiEritreaCDfilename = "BurundiEritreaCD.txt";
	std::ofstream BurundiEritreaCDfile;
	BurundiEritreaCDfile.open(BurundiEritreaCDfilename);
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
		BurundiEritreaCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Bosnia and Herzegovina vs Romania EM
	ci1 = CountryIndex(code, "BIH");
	ci2 = CountryIndex(code, "ROU");
	std::string BosniaandHerzegovinaRomaniafilename;
	BosniaandHerzegovinaRomaniafilename = "BosniaandHerzegovinaRomaniaEM.txt";
	std::ofstream BosniaandHerzegovinaRomaniafile;
	BosniaandHerzegovinaRomaniafile.open(BosniaandHerzegovinaRomaniafilename);
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
		BosniaandHerzegovinaRomaniafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Bulgaria vs Serbia EM
	ci1 = CountryIndex(code, "BGR");
	ci2 = CountryIndex(code, "SRB");
	std::string BulgariaSerbiafilename;
	BulgariaSerbiafilename = "BulgariaSerbiaEM.txt";
	std::ofstream BulgariaSerbiafile;
	BulgariaSerbiafile.open(BulgariaSerbiafilename);
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
		BulgariaSerbiafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Burundi vs Seychelles CD
	ci1 = CountryIndex(code, "BDI");
	ci2 = CountryIndex(code, "SYC");
	std::string BurundiSeychellesCDfilename;
	BurundiSeychellesCDfilename = "BurundiSeychellesCD.txt";
	std::ofstream BurundiSeychellesCDfile;
	BurundiSeychellesCDfile.open(BurundiSeychellesCDfilename);
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
		BurundiSeychellesCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Cambodia vs Vietnam CD
	ci1 = CountryIndex(code, "KHM");
	ci2 = CountryIndex(code, "VNM");
	std::string CambodiaVietnamCDfilename;
	CambodiaVietnamCDfilename = "CambodiaVietnamCD.txt";
	std::ofstream CambodiaVietnamCDfile;
	CambodiaVietnamCDfile.open(CambodiaVietnamCDfilename);
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
		CambodiaVietnamCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Canada vs Cuba EM
	ci1 = CountryIndex(code, "CAN");
	ci2 = CountryIndex(code, "CUB");
	std::string CanadaCubafilename;
	CanadaCubafilename = "CanadaCubaEM.txt";
	std::ofstream CanadaCubafile;
	CanadaCubafile.open(CanadaCubafilename);
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
		CanadaCubafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Canada vs Saint Vincent and the Grenadines EM
	ci1 = CountryIndex(code, "CAN");
	ci2 = CountryIndex(code, "VCT");
	std::string CanadaSaintVincentandtheGrenadinesfilename;
	CanadaSaintVincentandtheGrenadinesfilename = "CanadaSaintVincentandtheGrenadinesEM.txt";
	std::ofstream CanadaSaintVincentandtheGrenadinesfile;
	CanadaSaintVincentandtheGrenadinesfile.open(CanadaSaintVincentandtheGrenadinesfilename);
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
		CanadaSaintVincentandtheGrenadinesfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Cape Verde vs Tunisia EM
	ci1 = CountryIndex(code, "CPV");
	ci2 = CountryIndex(code, "TUN");
	std::string CapeVerdeTunisiafilename;
	CapeVerdeTunisiafilename = "CapeVerdeTunisiaEM.txt";
	std::ofstream CapeVerdeTunisiafile;
	CapeVerdeTunisiafile.open(CapeVerdeTunisiafilename);
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
		CapeVerdeTunisiafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Croatia vs Hungary CD
	ci1 = CountryIndex(code, "HRV");
	ci2 = CountryIndex(code, "HUN");
	std::string CroatiaHungaryCDfilename;
	CroatiaHungaryCDfilename = "CroatiaHungaryCD.txt";
	std::ofstream CroatiaHungaryCDfile;
	CroatiaHungaryCDfile.open(CroatiaHungaryCDfilename);
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
		CroatiaHungaryCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Croatia vs Hungary EM
	ci1 = CountryIndex(code, "HRV");
	ci2 = CountryIndex(code, "HUN");
	std::string CroatiaHungaryfilename;
	CroatiaHungaryfilename = "CroatiaHungaryEM.txt";
	std::ofstream CroatiaHungaryfile;
	CroatiaHungaryfile.open(CroatiaHungaryfilename);
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
		CroatiaHungaryfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Croatia vs Hungary _0_14 EM
	ci1 = CountryIndex(code, "HRV");
	ci2 = CountryIndex(code, "HUN");
	std::string CroatiaHungary_0_14filename;
	CroatiaHungary_0_14filename = "CroatiaHungary_0_14EM.txt";
	std::ofstream CroatiaHungary_0_14file;
	CroatiaHungary_0_14file.open(CroatiaHungary_0_14filename);
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
		CroatiaHungary_0_14file << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << HRVexcessmortalitypscore_0_14interpolated[d] << "\t" << HUNexcessmortalitypscore_0_14interpolated[d] << std::endl;
	}
	std::cout << ".";

	// Croatia vs Hungary _15_64 EM
	ci1 = CountryIndex(code, "HRV");
	ci2 = CountryIndex(code, "HUN");
	std::string CroatiaHungary_15_64filename;
	CroatiaHungary_15_64filename = "CroatiaHungary_15_64EM.txt";
	std::ofstream CroatiaHungary_15_64file;
	CroatiaHungary_15_64file.open(CroatiaHungary_15_64filename);
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
		CroatiaHungary_15_64file << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << HRVexcessmortalitypscore_15_64interpolated[d] << "\t" << HUNexcessmortalitypscore_15_64interpolated[d] << std::endl;
	}
	std::cout << ".";

	// Croatia vs Hungary _65_74 EM
	ci1 = CountryIndex(code, "HRV");
	ci2 = CountryIndex(code, "HUN");
	std::string CroatiaHungary_65_74filename;
	CroatiaHungary_65_74filename = "CroatiaHungary_65_74EM.txt";
	std::ofstream CroatiaHungary_65_74file;
	CroatiaHungary_65_74file.open(CroatiaHungary_65_74filename);
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
		CroatiaHungary_65_74file << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << HRVexcessmortalitypscore_65_74interpolated[d] << "\t" << HUNexcessmortalitypscore_65_74interpolated[d] << std::endl;
	}
	std::cout << ".";

	// Croatia vs Hungary _75_84 EM
	ci1 = CountryIndex(code, "HRV");
	ci2 = CountryIndex(code, "HUN");
	std::string CroatiaHungary_75_84filename;
	CroatiaHungary_75_84filename = "CroatiaHungary_75_84EM.txt";
	std::ofstream CroatiaHungary_75_84file;
	CroatiaHungary_75_84file.open(CroatiaHungary_75_84filename);
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
		CroatiaHungary_75_84file << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << HRVexcessmortalitypscore_75_84interpolated[d] << "\t" << HUNexcessmortalitypscore_75_84interpolated[d] << std::endl;
	}
	std::cout << ".";

	// Croatia vs Hungary _85p EM
	ci1 = CountryIndex(code, "HRV");
	ci2 = CountryIndex(code, "HUN");
	std::string CroatiaHungary_85pfilename;
	CroatiaHungary_85pfilename = "CroatiaHungary_85pEM.txt";
	std::ofstream CroatiaHungary_85pfile;
	CroatiaHungary_85pfile.open(CroatiaHungary_85pfilename);
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
		CroatiaHungary_85pfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << HRVexcessmortalitypscore_85pinterpolated[d] << "\t" << HUNexcessmortalitypscore_85pinterpolated[d] << std::endl;
	}
	std::cout << ".";

	// Cuba vs Jamaica CD
	ci1 = CountryIndex(code, "CUB");
	ci2 = CountryIndex(code, "JAM");
	std::string CubaJamaicaCDfilename;
	CubaJamaicaCDfilename = "CubaJamaicaCD.txt";
	std::ofstream CubaJamaicaCDfile;
	CubaJamaicaCDfile.open(CubaJamaicaCDfilename);
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
		CubaJamaicaCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Cuba vs Jamaica EM
	ci1 = CountryIndex(code, "CUB");
	ci2 = CountryIndex(code, "JAM");
	std::string CubaJamaicafilename;
	CubaJamaicafilename = "CubaJamaicaEM.txt";
	std::ofstream CubaJamaicafile;
	CubaJamaicafile.open(CubaJamaicafilename);
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
		CubaJamaicafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Cuba vs Saint Vincent and the Grenadines EM
	ci1 = CountryIndex(code, "CUB");
	ci2 = CountryIndex(code, "VCT");
	std::string CubaSaintVincentandtheGrenadinesfilename;
	CubaSaintVincentandtheGrenadinesfilename = "CubaSaintVincentandtheGrenadinesEM.txt";
	std::ofstream CubaSaintVincentandtheGrenadinesfile;
	CubaSaintVincentandtheGrenadinesfile.open(CubaSaintVincentandtheGrenadinesfilename);
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
		CubaSaintVincentandtheGrenadinesfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Denmark vs Finland EM
	ci1 = CountryIndex(code, "DNK");
	ci2 = CountryIndex(code, "FIN");
	std::string DenmarkFinlandfilename;
	DenmarkFinlandfilename = "DenmarkFinlandEM.txt";
	std::ofstream DenmarkFinlandfile;
	DenmarkFinlandfile.open(DenmarkFinlandfilename);
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
		DenmarkFinlandfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Denmark vs Germany EM
	ci1 = CountryIndex(code, "DNK");
	ci2 = CountryIndex(code, "DEU");
	std::string DenmarkGermanyfilename;
	DenmarkGermanyfilename = "DenmarkGermanyEM.txt";
	std::ofstream DenmarkGermanyfile;
	DenmarkGermanyfile.open(DenmarkGermanyfilename);
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
		DenmarkGermanyfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Denmark vs Norway CD
	ci1 = CountryIndex(code, "DNK");
	ci2 = CountryIndex(code, "NOR");
	std::string DenmarkNorwayCDfilename;
	DenmarkNorwayCDfilename = "DenmarkNorwayCD.txt";
	std::ofstream DenmarkNorwayCDfile;
	DenmarkNorwayCDfile.open(DenmarkNorwayCDfilename);
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
		DenmarkNorwayCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Denmark vs Norway EM
	ci1 = CountryIndex(code, "DNK");
	ci2 = CountryIndex(code, "NOR");
	std::string DenmarkNorwayfilename;
	DenmarkNorwayfilename = "DenmarkNorwayEM.txt";
	std::ofstream DenmarkNorwayfile;
	DenmarkNorwayfile.open(DenmarkNorwayfilename);
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
		DenmarkNorwayfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Dominican Republic vs Haiti CD
	ci1 = CountryIndex(code, "DOM");
	ci2 = CountryIndex(code, "HTI");
	std::string DominicanRepublicHaiticdfilename;
	DominicanRepublicHaiticdfilename = "DominicanRepublicHaitiCD.txt";
	std::ofstream DominicanRepublicHaiticdfile;
	DominicanRepublicHaiticdfile.open(DominicanRepublicHaiticdfilename);
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
		DominicanRepublicHaiticdfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// FaroeIslands vs Norway CD
	ci1 = CountryIndex(code, "FRO");
	ci2 = CountryIndex(code, "NOR");
	std::string FaroeIslandsNorwayCDfilename;
	FaroeIslandsNorwayCDfilename = "FaroeIslandsNorwayCD.txt";
	std::ofstream FaroeIslandsNorwayCDfile;
	FaroeIslandsNorwayCDfile.open(FaroeIslandsNorwayCDfilename);
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
		FaroeIslandsNorwayCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// FaroeIslands vs Norway EM
	ci1 = CountryIndex(code, "FRO");
	ci2 = CountryIndex(code, "NOR");
	std::string FaroeIslandsNorwayfilename;
	FaroeIslandsNorwayfilename = "FaroeIslandsNorwayEM.txt";
	std::ofstream FaroeIslandsNorwayfile;
	FaroeIslandsNorwayfile.open(FaroeIslandsNorwayfilename);
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
		FaroeIslandsNorwayfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// French Polynesia vs New Zealand EM
	ci1 = CountryIndex(code, "PYF");
	ci2 = CountryIndex(code, "NZL");
	std::string FrenchPolynesiaNewZealandfilename;
	FrenchPolynesiaNewZealandfilename = "FrenchPolynesiaNewZealandEM.txt";
	std::ofstream FrenchPolynesiaNewZealandfile;
	FrenchPolynesiaNewZealandfile.open(FrenchPolynesiaNewZealandfilename);
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
		FrenchPolynesiaNewZealandfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Gibraltar vs India CD
	ci1 = CountryIndex(code, "GIB");
	ci2 = CountryIndex(code, "IND");
	std::string GibraltarIndiafilename;
	GibraltarIndiafilename = "GibraltarIndiaCD.txt";
	std::ofstream GibraltarIndiafile;
	GibraltarIndiafile.open(GibraltarIndiafilename);
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
		GibraltarIndiafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Greece vs Serbia EM
	ci1 = CountryIndex(code, "GRC");
	ci2 = CountryIndex(code, "SRB");
	std::string GreeceSerbiafilename;
	GreeceSerbiafilename = "GreeceSerbiaEM.txt";
	std::ofstream GreeceSerbiafile;
	GreeceSerbiafile.open(GreeceSerbiafilename);
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
		GreeceSerbiafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Hong Kong vs Macao EM
	ci1 = CountryIndex(code, "HKG");
	ci2 = CountryIndex(code, "MAC");
	std::string HongKongMacaofilename;
	HongKongMacaofilename = "HongKongMacaoEM.txt";
	std::ofstream HongKongMacaofile;
	HongKongMacaofile.open(HongKongMacaofilename);
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
		HongKongMacaofile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Hong Kong vs Philippines CD
	ci1 = CountryIndex(code, "HKG");
	ci2 = CountryIndex(code, "PHL");
	std::string HongKongPhilippinesCDfilename;
	HongKongPhilippinesCDfilename = "HongKongPhilippinesCD.txt";
	std::ofstream HongKongPhilippinesCDfile;
	HongKongPhilippinesCDfile.open(HongKongPhilippinesCDfilename);
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
		HongKongPhilippinesCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Hong Kong vs Philippines EM
	ci1 = CountryIndex(code, "HKG");
	ci2 = CountryIndex(code, "PHL");
	std::string HongKongPhilippinesfilename;
	HongKongPhilippinesfilename = "HongKongPhilippinesEM.txt";
	std::ofstream HongKongPhilippinesfile;
	HongKongPhilippinesfile.open(HongKongPhilippinesfilename);
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
		HongKongPhilippinesfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Hong Kong vs South Korea EM
	ci1 = CountryIndex(code, "HKG");
	ci2 = CountryIndex(code, "KOR");
	std::string HongKongSouthKoreafilename;
	HongKongSouthKoreafilename = "HongKongSouthKoreaEM.txt";
	std::ofstream HongKongSouthKoreafile;
	HongKongSouthKoreafile.open(HongKongSouthKoreafilename);
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
		HongKongSouthKoreafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Hungary vs Romania CD
	ci1 = CountryIndex(code, "HUN");
	ci2 = CountryIndex(code, "ROU");
	std::string HungaryRomaniaCDfilename;
	HungaryRomaniaCDfilename = "HungaryRomaniaCD.txt";
	std::ofstream HungaryRomaniaCDfile;
	HungaryRomaniaCDfile.open(HungaryRomaniaCDfilename);
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
		HungaryRomaniaCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Hungary vs Romania EM
	ci1 = CountryIndex(code, "HUN");
	ci2 = CountryIndex(code, "ROU");
	std::string HungaryRomaniafilename;
	HungaryRomaniafilename = "HungaryRomaniaEM.txt";
	std::ofstream HungaryRomaniafile;
	HungaryRomaniafile.open(HungaryRomaniafilename);
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
		HungaryRomaniafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Indonesia vs Philippines CD
	ci1 = CountryIndex(code, "IDN");
	ci2 = CountryIndex(code, "PHL");
	std::string IndonesiaPhilippinesfilename;
	IndonesiaPhilippinesfilename = "IndonesiaPhilippinesCD.txt";
	std::ofstream IndonesiaPhilippinesfile;
	IndonesiaPhilippinesfile.open(IndonesiaPhilippinesfilename);
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
		IndonesiaPhilippinesfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Iran vs Iraq CD
	ci1 = CountryIndex(code, "IRN");
	ci2 = CountryIndex(code, "IRQ");
	std::string IranIraqfilename;
	IranIraqfilename = "IranIraqCD.txt";
	std::ofstream IranIraqfile;
	IranIraqfile.open(IranIraqfilename);
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
		IranIraqfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Iraq vs Oman CD
	ci1 = CountryIndex(code, "IRQ");
	ci2 = CountryIndex(code, "OMN");
	std::string IraqOmanfilename;
	IraqOmanfilename = "IraqOmanCD.txt";
	std::ofstream IraqOmanfile;
	IraqOmanfile.open(IraqOmanfilename);
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
		IraqOmanfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Jamaica vs Saint Vincent and the Grenadines CD
	ci1 = CountryIndex(code, "JAM");
	ci2 = CountryIndex(code, "VCT");
	std::string JamaicaSaintVincentandtheGrenadinesCDfilename;
	JamaicaSaintVincentandtheGrenadinesCDfilename = "JamaicaSaintVincentandtheGrenadinesCD.txt";
	std::ofstream JamaicaSaintVincentandtheGrenadinesCDfile;
	JamaicaSaintVincentandtheGrenadinesCDfile.open(JamaicaSaintVincentandtheGrenadinesCDfilename);
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
		JamaicaSaintVincentandtheGrenadinesCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Jamaica vs Saint Vincent and the Grenadines EM
	ci1 = CountryIndex(code, "JAM");
	ci2 = CountryIndex(code, "VCT");
	std::string JamaicaSaintVincentandtheGrenadinesfilename;
	JamaicaSaintVincentandtheGrenadinesfilename = "JamaicaSaintVincentandtheGrenadinesEM.txt";
	std::ofstream JamaicaSaintVincentandtheGrenadinesfile;
	JamaicaSaintVincentandtheGrenadinesfile.open(JamaicaSaintVincentandtheGrenadinesfilename);
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
		JamaicaSaintVincentandtheGrenadinesfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Japan vs Malaysia EM
	ci1 = CountryIndex(code, "JPN");
	ci2 = CountryIndex(code, "MYS");
	std::string JapanMalaysiafilename;
	JapanMalaysiafilename = "JapanMalaysiaEM.txt";
	std::ofstream JapanMalaysiafile;
	JapanMalaysiafile.open(JapanMalaysiafilename);
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
		JapanMalaysiafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Japan vs Philippines EM
	ci1 = CountryIndex(code, "JPN");
	ci2 = CountryIndex(code, "PHL");
	std::string JapanPhilippinesfilename;
	JapanPhilippinesfilename = "JapanPhilippinesEM.txt";
	std::ofstream JapanPhilippinesfile;
	JapanPhilippinesfile.open(JapanPhilippinesfilename);
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
		JapanPhilippinesfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Japan vs South Korea EM
	ci1 = CountryIndex(code, "JPN");
	ci2 = CountryIndex(code, "KOR");
	std::string JapanSouthKoreafilename;
	JapanSouthKoreafilename = "JapanSouthKoreaEM.txt";
	std::ofstream JapanSouthKoreafile;
	JapanSouthKoreafile.open(JapanSouthKoreafilename);
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
		JapanSouthKoreafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Kyrgyzstan vs Uzbekistan CD
	ci1 = CountryIndex(code, "KGZ");
	ci2 = CountryIndex(code, "UZB");
	std::string KyrgyzstanUzbekistanCDfilename;
	KyrgyzstanUzbekistanCDfilename = "KyrgyzstanUzbekistanCD.txt";
	std::ofstream KyrgyzstanUzbekistanCDfile;
	KyrgyzstanUzbekistanCDfile.open(KyrgyzstanUzbekistanCDfilename);
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
		KyrgyzstanUzbekistanCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Kyrgyzstan vs Uzbekistan EM
	ci1 = CountryIndex(code, "KGZ");
	ci2 = CountryIndex(code, "UZB");
	std::string KyrgyzstanUzbekistanfilename;
	KyrgyzstanUzbekistanfilename = "KyrgyzstanUzbekistanEM.txt";
	std::ofstream KyrgyzstanUzbekistanfile;
	KyrgyzstanUzbekistanfile.open(KyrgyzstanUzbekistanfilename);
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
		KyrgyzstanUzbekistanfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Latvia vs Ukraine CD
	ci1 = CountryIndex(code, "LVA");
	ci2 = CountryIndex(code, "UKR");
	std::string LatviaUkraineCDfilename;
	LatviaUkraineCDfilename = "LatviaUkraineCD.txt";
	std::ofstream LatviaUkraineCDfile;
	LatviaUkraineCDfile.open(LatviaUkraineCDfilename);
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
		LatviaUkraineCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Latvia vs Ukraine EM
	ci1 = CountryIndex(code, "LVA");
	ci2 = CountryIndex(code, "UKR");
	std::string LatviaUkrainefilename;
	LatviaUkrainefilename = "LatviaUkraineEM.txt";
	std::ofstream LatviaUkrainefile;
	LatviaUkrainefile.open(LatviaUkrainefilename);
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
		LatviaUkrainefile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Lebanon vs Palestine EM
	ci1 = CountryIndex(code, "LBN");
	ci2 = CountryIndex(code, "PSE");
	std::string LebanonPalestinefilename;
	LebanonPalestinefilename = "LebanonPalestineEM.txt";
	std::ofstream LebanonPalestinefile;
	LebanonPalestinefile.open(LebanonPalestinefilename);
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
		LebanonPalestinefile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Malaysia vs Mongolia EM
	ci1 = CountryIndex(code, "MYS");
	ci2 = CountryIndex(code, "MNG");
	std::string MalaysiaMongoliafilename;
	MalaysiaMongoliafilename = "MalaysiaMongoliaEM.txt";
	std::ofstream MalaysiaMongoliafile;
	MalaysiaMongoliafile.open(MalaysiaMongoliafilename);
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
		MalaysiaMongoliafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Malaysia vs Taiwan EM
	ci1 = CountryIndex(code, "MYS");
	ci2 = CountryIndex(code, "TWN");
	std::string MalaysiaTaiwanfilename;
	MalaysiaTaiwanfilename = "MalaysiaTaiwanEM.txt";
	std::ofstream MalaysiaTaiwanfile;
	MalaysiaTaiwanfile.open(MalaysiaTaiwanfilename);
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
		MalaysiaTaiwanfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Malaysia vs Thailand CD
	ci1 = CountryIndex(code, "MYS");
	ci2 = CountryIndex(code, "THA");
	std::string MalaysiaThailandCDfilename;
	MalaysiaThailandCDfilename = "MalaysiaThailandCD.txt";
	std::ofstream MalaysiaThailandCDfile;
	MalaysiaThailandCDfile.open(MalaysiaThailandCDfilename);
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
		MalaysiaThailandCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Malaysia vs Thailand EM
	ci1 = CountryIndex(code, "MYS");
	ci2 = CountryIndex(code, "THA");
	std::string MalaysiaThailandfilename;
	MalaysiaThailandfilename = "MalaysiaThailandEM.txt";
	std::ofstream MalaysiaThailandfile;
	MalaysiaThailandfile.open(MalaysiaThailandfilename);
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
		MalaysiaThailandfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Mauritius vs Seychelles CD
	ci1 = CountryIndex(code, "MUS");
	ci2 = CountryIndex(code, "SYC");
	std::string MauritiusSeychellesCDfilename;
	MauritiusSeychellesCDfilename = "MauritiusSeychellesCD.txt";
	std::ofstream MauritiusSeychellesCDfile;
	MauritiusSeychellesCDfile.open(MauritiusSeychellesCDfilename);
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
		MauritiusSeychellesCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Mauritius vs Seychelles EM
	ci1 = CountryIndex(code, "MUS");
	ci2 = CountryIndex(code, "SYC");
	std::string MauritiusSeychellesfilename;
	MauritiusSeychellesfilename = "MauritiusSeychellesEM.txt";
	std::ofstream MauritiusSeychellesfile;
	MauritiusSeychellesfile.open(MauritiusSeychellesfilename);
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
		MauritiusSeychellesfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Mongolia vs Thailand CD
	ci1 = CountryIndex(code, "MNG");
	ci2 = CountryIndex(code, "THA");
	std::string MongoliaThailandCDfilename;
	MongoliaThailandCDfilename = "MongoliaThailandCD.txt";
	std::ofstream MongoliaThailandCDfile;
	MongoliaThailandCDfile.open(MongoliaThailandCDfilename);
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
		MongoliaThailandCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Mongolia vs Thailand EM
	ci1 = CountryIndex(code, "MNG");
	ci2 = CountryIndex(code, "THA");
	std::string MongoliaThailandfilename;
	MongoliaThailandfilename = "MongoliaThailandEM.txt";
	std::ofstream MongoliaThailandfile;
	MongoliaThailandfile.open(MongoliaThailandfilename);
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
		MongoliaThailandfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Papua New Guinea vs Solomon Islands CD
	ci1 = CountryIndex(code, "PNG");
	ci2 = CountryIndex(code, "SLB");
	std::string PapuaNewGuineaSolomonIslandsCDfilename;
	PapuaNewGuineaSolomonIslandsCDfilename = "PapuaNewGuineaSolomonIslandsCD.txt";
	std::ofstream PapuaNewGuineaSolomonIslandsCDfile;
	PapuaNewGuineaSolomonIslandsCDfile.open(PapuaNewGuineaSolomonIslandsCDfilename);
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
		PapuaNewGuineaSolomonIslandsCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Philippines vs South Korea EM
	ci1 = CountryIndex(code, "PHL");
	ci2 = CountryIndex(code, "KOR");
	std::string PhilippinesSouthKoreafilename;
	PhilippinesSouthKoreafilename = "PhilippinesSouthKoreaEM.txt";
	std::ofstream PhilippinesSouthKoreafile;
	PhilippinesSouthKoreafile.open(PhilippinesSouthKoreafilename);
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
		PhilippinesSouthKoreafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Romania vs Russia CD
	ci1 = CountryIndex(code, "ROU");
	ci2 = CountryIndex(code, "RUS");
	std::string RomaniaRussiaCDfilename;
	RomaniaRussiaCDfilename = "RomaniaRussiaCD.txt";
	std::ofstream RomaniaRussiaCDfile;
	RomaniaRussiaCDfile.open(RomaniaRussiaCDfilename);
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
		RomaniaRussiaCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Romania vs Russia EM
	ci1 = CountryIndex(code, "ROU");
	ci2 = CountryIndex(code, "RUS");
	std::string RomaniaRussiafilename;
	RomaniaRussiafilename = "RomaniaRussiaEM.txt";
	std::ofstream RomaniaRussiafile;
	RomaniaRussiafile.open(RomaniaRussiafilename);
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
		RomaniaRussiafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Rwanda vs Uganda CD
	ci1 = CountryIndex(code, "RWA");
	ci2 = CountryIndex(code, "UGA");
	std::string RwandaUgandafilename;
	RwandaUgandafilename = "RwandaUgandaCD.txt";
	std::ofstream RwandaUgandafile;
	RwandaUgandafile.open(RwandaUgandafilename);
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
		RwandaUgandafile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Seychelles vs Tanzania CD
	ci1 = CountryIndex(code, "SYC");
	ci2 = CountryIndex(code, "TZA");
	std::string SeychellesTanzaniaCDfilename;
	SeychellesTanzaniaCDfilename = "SeychellesTanzaniaCD.txt";
	std::ofstream SeychellesTanzaniaCDfile;
	SeychellesTanzaniaCDfile.open(SeychellesTanzaniaCDfilename);
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
		SeychellesTanzaniaCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Singapore vs Thailand CD
	ci1 = CountryIndex(code, "SGP");
	ci2 = CountryIndex(code, "THA");
	std::string SingaporeThailandCDfilename;
	SingaporeThailandCDfilename = "SingaporeThailandCD.txt";
	std::ofstream SingaporeThailandCDfile;
	SingaporeThailandCDfile.open(SingaporeThailandCDfilename);
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
		SingaporeThailandCDfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Singapore vs Thailand EM
	ci1 = CountryIndex(code, "SGP");
	ci2 = CountryIndex(code, "THA");
	std::string SingaporeThailandfilename;
	SingaporeThailandfilename = "SingaporeThailandEM.txt";
	std::ofstream SingaporeThailandfile;
	SingaporeThailandfile.open(SingaporeThailandfilename);
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
		SingaporeThailandfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Sweden vs United Kingdom lockdown
	ci1 = CountryIndex(code, "SWE");
	ci2 = CountryIndex(code, "GBR");
	std::string SWEGBRfilename;
		SWEGBRfilename = "SWEGBR.txt";
	std::ofstream SWEGBRfile;
	SWEGBRfile.open(SWEGBRfilename);
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
		SWEGBRfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << lockdown[ci1][d] << "\t" << lockdown[ci2][d] << "\t" << excessmortalitypscoreinterpolated[ci1][d] << "\t" << excessmortalitypscoreinterpolated[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Thailand vs Vietnam CD
	ci1 = CountryIndex(code, "THA");
	ci2 = CountryIndex(code, "VNM");
	std::string ThailandVietnamfilename;
	ThailandVietnamfilename = "ThailandVietnamCD.txt";
	std::ofstream ThailandVietnamfile;
	ThailandVietnamfile.open(ThailandVietnamfilename);
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
		ThailandVietnamfile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";

	// Zambia vs Zimbabwe CD
	ci1 = CountryIndex(code, "ZMB");
	ci2 = CountryIndex(code, "ZWE");
	std::string ZambiaZimbabwefilename;
	ZambiaZimbabwefilename = "ZambiaZimbabweCD.txt";
	std::ofstream ZambiaZimbabwefile;
	ZambiaZimbabwefile.open(ZambiaZimbabwefilename);
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
		ZambiaZimbabwefile << dp << d2.day() << "/" << mp << d2.month().as_number() << "/" << d2.year() << "\t" << vaccinations[ci1][d] << "\t" << vaccinations[ci2][d] << "\t" << coviddeaths[ci1][d] << "\t" << coviddeaths[ci2][d] << std::endl;
	}
	std::cout << ".";




	std::string vaxdiff, cddiff, emdiff;

	for (int c1 = 0; c1 < numentities; c1++)
		for (int c2 = 0; c2 < numentities; c2++) {
			// for each pair of countries, c1 and c2

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
				if (isnan(v1[i][1]) || isnan(v2[i][1])) {
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
				if (isnan(cd1[i][1]) || isnan(cd2[i][1])) {
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
				em1[d1][1] = excessmortalitypscoreinterpolated[c1][d1];
				em2[d1][0] = d1;
				em2[d1][1] = excessmortalitypscoreinterpolated[c2][d1];
			}
			for (unsigned int i = 0; i < em1.size(); ) {
				if (isnan(em1[i][1]) || isnan(em2[i][1])) {
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
	outputfile << "Continent" <<  "\t" << "Country1" << "\t" << "Country2" << "\t" << "vax p_norm" << "\t" << "cd p_norm" << "\t" << "em p_norm" << std::endl;
	for (unsigned int i = 0; i < display.size(); i++)
		outputfile << display[i] << std::endl;

	std::cout << std::endl;
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;
	double mins = floor(diff.count() / 60);
	double secs = diff.count() - mins*60;
	//std::cout << "Elapsed time: " << mins << " minutes " << secs << " seconds" << std::endl;
	std::cout << "Finished!" << std::endl;
	std::cout << "5) Copy this program's output files to the directory containing SIR.py." << std::endl;
	std::cout << "6) Run SIR.py." << std::endl;
	std::cin.get();
	return 0;
}