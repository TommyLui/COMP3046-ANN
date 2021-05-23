#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <time.h>
#include <windows.h>
#include <algorithm>
#include "ANN.h"
#include "OpenMPANN.h"

#include <omp.h>

using namespace std;

int main(int argc, const char* argv[]) {

	vector< vector<float> > X_train;
	vector<float> y_train;

	ifstream myfile("train_small.txt");

	if (myfile.is_open())
	{
		cout << "Loading data ...\n";
		string line;
		while (getline(myfile, line))
		{
			int x, y;
			vector<float> X;
			stringstream ss(line);
			ss >> y;
			y_train.push_back(y);
			for (int i = 0; i < 28 * 28; i++) {
				ss >> x;
				X.push_back(x / 255.0);
			}
			X_train.push_back(X);
		}

		myfile.close();
		cout << "Loading data finished.\n";
	}
	else
		cout << "Unable to open file" << '\n';



	//class & train
	ANN ANN;
	OpenMPANN PANN;

	clock_t start, end;
	double Pstart, Pend;

	//Set 1:learning rate, 2:batch size, 3:stop epoch
	ANN.trainSet(0.1, 32, 5);
	PANN.trainSet(0.1, 32, 5);

	//Set 1:input, 2:hidden layer numbers, 3:node numbers
	ANN.weightSet(X_train, 2, 5);
	PANN.weightSet(X_train, 2, 5);

	//Load Weight
	ANN.loadWeight("weight1.txt");
	PANN.loadWeight("weight1.txt");

	//Start
	start = clock();
	ANN.train(X_train, y_train);
	end = clock() - start;

	PANN.threadSet(8);
	Pstart = omp_get_wtime();
	PANN.train(X_train, y_train);
	Pend = omp_get_wtime() - Pstart;

	cout << "Without openMP Time: " << (float)end / CLOCKS_PER_SEC << endl;
	cout << "Multi-thread Time: " << Pend << endl;

	//end

	return 0;
}