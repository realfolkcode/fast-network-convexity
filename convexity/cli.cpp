#include "algo.h"

using namespace std;

int main(int argc, char* argv[]) {
	string fn_in;
	string fn_out = "out.txt";
	string fn_dist;
	int k;
	int repeats = 10;
	//parse args
	if (argc > 1) {
		fn_in = argv[1];
		cout << fn_in << endl;
		if (argc > 2) {
			sscanf(argv[2], "%d", &k);
			cout << "k = " << k << endl;
			if (argc > 3) {
				sscanf(argv[3], "%d", &repeats);
				cout << "repeats = " << repeats << endl;
				if (argc > 4) {
					fn_out = argv[4];
					cout << fn_out << endl;
				}
			}
		}
		else {
			cout << "Second argument should be number of elements!" << endl;
		}
	}
	else {
		cout << "First argument should be path to pajek file containing a network!" << endl;
		exit(0);
	}

	auto net = readPajek(fn_in);
	cout << "Network loaded." << endl;
	cout << "Nodes in the largest connected component: " << net.size() << endl;

	fn_dist = fn_out + ".dist";
	fn_out = fn_out + ".out";
	std::ofstream output(fn_out);
	output << repeats << endl;

	vector<vector<int>> bases = generateBases(fn_in, k, repeats);
	for (auto b : bases) {
		for (auto elem : b) {
			output << elem << ' ';
		}
		output << endl;
	}
	cout << "Bases have been sampled" << endl;

	std::ofstream dist_out(fn_dist);
	auto dists = distances(net);
	for (int i = 0; i < dists.size(); ++i) {
		for (int j = 0; j < dists[i].size(); ++j) {
			dist_out << dists[i][j] << " ";
		}
		dist_out << endl;
	}
	cout << "Distances have been calculated" << endl;

	int i = 0;
	for (auto b : bases) {
		SubGraph s(net);
		auto hull = convexHull(net, dists, s, b);
		for (auto elem : hull) {
			output << elem << ' ';
		}
		output << endl;
		cout << "Hull: " <<i << endl;
		++i;
	}

	output.close();
	return 0;
}