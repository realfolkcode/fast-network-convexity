#define _CRT_SECURE_NO_WARNINGS
#include <vector>
#include <deque>
#include <fstream>
#include <cstdio>
#include <utility>
#include <unordered_set>
#include <random>
#include <cassert>
#include <numeric>
#include <iostream>
#include <chrono>
#include <cstring>
#include <climits>
#include <iterator>
#include <set>
#include "algo.h"

using namespace std;
#define NO_VALUE -1

SubGraph::SubGraph(const vector<vector<int>>& network) :
	present(vector<char>(network.size()))
{}
inline bool SubGraph::insert(int vertex) {
	bool inserted = !present[vertex];
	if (inserted) {
		present[vertex] = 1;
		list.push_back(vertex);
	}
	return inserted;
}

template <typename T>
bool contains(const vector<T>& vec, T& el) {
	for (const T& i : vec) {
		if (i == el) {
			return true;
		}
	}
	return false;
}

vector<vector<int>> readPajek(string fn, vector<string>* names) {
	ifstream input(fn, ifstream::in);
	if (!input.is_open()) {
		perror(fn.c_str());
		exit(0);
	}
	vector<vector<int>> res;
	char line[1024];
	input.getline(line, 1024);
	int n = 0;
	for (; ; n++) {
		input.getline(line, 1024);
		if (line[0] == '*') {
			break;
		}
		if (names != nullptr) {
			names->push_back(line);
		}
	}
	res.resize(n, vector<int>());
	int m = 0;
	int a, b;
	for (; ; m++) {
		input.getline(line, 256);
		if (sscanf(line, "%d %d", &a, &b) == EOF) {
			break;
		}
		a--; // pajek uses 1-based indexing
		b--;
		if (!contains(res[a], b)) {
			res[a].push_back(b);
		}
		if (!contains(res[b], a)) {
			res[b].push_back(a);
		}
	}
	return res;
}

vector<vector<int>> distances(const vector<vector<int>>& network) {
	vector<vector<int>> res(network.size());
	//res.reserve(network.size());
#pragma omp parallel for
	for (int vertex = 0; vertex < network.size(); vertex++) {
		vector<int> distances(network.size(), NO_VALUE);
		distances[vertex] = 0;
		deque<int> todo;
		todo.push_back(vertex);
		while (!todo.empty()) {
			int current = todo.front();
			todo.pop_front();
			for (int neighbor : network[current]) {
				if (distances[neighbor] == NO_VALUE) {
					distances[neighbor] = distances[current] + 1;
					todo.push_back(neighbor);
				}
			}
		}
		res[vertex] = move(distances);
	}
	return res;
}

vector<int> convexHull(const vector<vector<int>>& network, const vector<vector<int>>& distances, SubGraph& subGraph, vector<int> base) {
	deque<int> todo;
	vector<int> insertions;
	for (auto newVertex : base) {
		todo.push_back(newVertex);
		insertions.push_back(newVertex);
		subGraph.insert(newVertex);
	}
	while (!todo.empty()) {
		int current = todo.front();
		todo.pop_front();
		for (int neighbor : network[current]) {
			if (!subGraph.present[neighbor]) {
				for (int endVertex : subGraph.list) {
					if (distances[current][endVertex] >= distances[current][neighbor] + distances[neighbor][endVertex]) {
						todo.push_back(neighbor);
						insertions.push_back(neighbor);
						subGraph.insert(neighbor);
						break;
					}
					else {
					}
				}
			}
		}
	}

	return insertions;
}

vector<vector<int>> generateBases(string fn, int k, int repeats) {
	auto net = readPajek(fn);
	vector<vector<int>> bases(repeats);
	vector<int> nodes(net.size());
	long long rnd_init = 14994518116208229;
	std::default_random_engine generator(rnd_init);
	for (int i = 0; i < net.size(); ++i) {
		nodes[i] = i;
	}
	for (int i = 0; i < repeats; ++i) {
		vector<int> s;
		std::sample(nodes.begin(), nodes.end(), std::back_inserter(s), k, generator);
		bases[i] = s;
	}
	return bases;
}