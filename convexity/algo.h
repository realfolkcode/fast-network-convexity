#pragma once
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

using namespace std;
#define NO_VALUE -1

class SubGraph {
public:
	vector<char> present;
	vector<int> list;
	SubGraph(const vector<vector<int>>& network);
	bool insert(int vertex);
};

template <typename T>
bool contains(const vector<T>& vec, T& el);

vector<vector<int>> readPajek(string fn, vector<string>* names = nullptr);

vector<vector<int>> distances(const vector<vector<int>>& network);

vector<int> convexHull(const vector<vector<int>>& network, const vector<vector<int>>& distances, SubGraph& subGraph, vector<int> base);

vector<vector<int>> generateBases(string fn, int k, int repeats);