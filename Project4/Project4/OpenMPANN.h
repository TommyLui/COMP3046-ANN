#ifndef OpenMPANN_H
#define OpenMPANN_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <cstdlib> 
#include <chrono>

#include <omp.h>

using namespace std;


class OpenMPANN
{
protected:
	int threadNum = 2;

	vector<vector<float>> X;
	vector<float> y;
	float learningRate;
	int batchSize;
	int stopEpochs;
	vector<float> weight;
	int layerNum;
	int nodeNum;
	vector<int> predict;
	vector< vector<float> > firstWeight;
	vector<float> firstBias;
	vector < vector < vector<float> >> otherWeight;
	vector< vector<float> > otherBias;
	vector< vector<float> > lastWeight;
	vector<float> lastBias;
	vector< vector<float> > hiddenLayer;
	vector < vector < vector<float> >> dataSet;
	vector < vector < vector<float> >> hiddenLayer_F;
	vector <vector<float> > ansSet;
	vector <vector<float> > feedForOut;
	vector <vector<float> > errorOfHiddenLayer;
public:
	OpenMPANN();

	void threadSet(int num);

	//func 1
	void weightSet(vector< vector<float>> X, int numOfHiddenLayer, int numOfNode);
	vector<int> feedForward(vector< vector<float>> X);
	//func 2
	void trainSet(float r, int b, int s);
	void train(vector< vector<float> > X, vector<float> y);
	//func 3
	void inference(vector<float> X);

	//func4
	void storeWeight(string fileName);
	void loadWeight(string fileName);

private:
	float randomWeight();
	float sigmoid(float input);
	float dsigmoid(float in);
	vector<float> scalarV(vector<float> V, float Snum);
	vector < vector<float>> substract(vector< vector<float> > X, vector< vector<float> > Y);
	vector < vector<float>> transpose(vector<vector<float>>& m);
	vector < vector<float>> m_m(vector< vector<float> > X, vector< vector<float> > Y);
	vector<float> m_v(vector< vector<float> > X, vector<float> Y);
	vector<float> v_v(vector<float> V1, vector<float> V2);
	vector<vector<float> >v_v_m(vector<float> V1, vector<float> V2);
};

OpenMPANN::OpenMPANN() {
}

void OpenMPANN::threadSet(int num) {
	this->threadNum = num;
}

void OpenMPANN::trainSet(float r, int b, int s) {
	this->learningRate = r;
	this->batchSize = b;
	this->stopEpochs = s;
}

void OpenMPANN::weightSet(vector< vector<float> > X, int numOfHiddenLayer, int numOfNode) {
	this->layerNum = numOfHiddenLayer;
	this->nodeNum = numOfNode;

	firstWeight.resize(X.size(), vector<float>(X[0].size()));
	firstBias.resize(X[0].size());
	otherWeight.resize((layerNum - 1), vector <vector<float>>(nodeNum, vector<float>(nodeNum)));
	otherBias.resize((layerNum - 1), vector<float>(nodeNum));
	lastWeight.resize(10, vector<float>(nodeNum));
	lastBias.resize(10);
	hiddenLayer.resize(layerNum, vector<float>(nodeNum));
	hiddenLayer_F.resize(batchSize, vector <vector<float>>(layerNum, vector<float>(nodeNum)));

	//Weight after input
	for (int i = 0; i < X.size(); i++) {
		for (int j = 0; j < X[0].size(); j++)
		{
			firstWeight[i][j] = randomWeight();
		}
	}


	//Bias after input
	for (int i = 0; i < X[0].size(); i++)
	{
		firstBias[i] = randomWeight();
	}

	//Weight after hiddenLayer1 to hiddenLayer Last
	for (int i = 0; i < layerNum - 1; i++) {
		for (int j = 0; j < nodeNum; j++) {
			for (int k = 0; k < nodeNum; k++)
			{
				otherWeight[i][j][k] = randomWeight();
			}
		}
	}

	//weight for output
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < nodeNum; j++)
		{
			lastWeight[i][j] = randomWeight();
		}
	}

	//for (int i = 0; i < layerNum; i++) testing(otherWeight[i]);

	//Bias after hiddenLayer1 to hiddenLayer Last
	for (int i = 0; i < layerNum - 1; i++) {
		for (int j = 0; j < nodeNum; j++)
		{
			otherBias[i][j] = randomWeight();
		}
	}
	for (int i = 0; i < 10; i++)
	{
		lastBias[i] = randomWeight();
	}
	//testing(otherBias);
}

float OpenMPANN::sigmoid(float toSigmoid) {
	return (1 / (1 + exp(-toSigmoid)));
}

float OpenMPANN::dsigmoid(float Sig) {
	return Sig * (1 - Sig);
}

float OpenMPANN::randomWeight() {
	return (2 * (float)rand() / (RAND_MAX)-1);;
}

vector<float> OpenMPANN::scalarV(vector<float> V, float Snum) {
	for (int i = 0; i < V.size(); i++) {
		V[i] = V[i] * Snum;
		//cout << V[i];
	}
	return V;
}

vector<float> OpenMPANN::v_v(vector<float> V1, vector<float> V2) {  //vector multipication (vector)
	vector<float> Mat;
	Mat.resize(V1.size());
	for (int i = 0; i < V1.size(); i++) {
		//for (int j = 0; j < V2.size(); j++) {
		Mat[i] = V1[i] * V2[i];
		//}
	}
	return Mat;
}

vector<vector<float> > OpenMPANN::v_v_m(vector<float> V1, vector<float> V2) {  //vector multipication (output matrix)
	vector<vector<float>> Mat;
	Mat.resize(V1.size(), vector<float>(V2.size()));

	int i;
	int j;

//#  pragma omp parallel for num_threads(threadNum) \
   default(shared) private(i,j)

	for (i = 0; i < V1.size(); i++) {
		for (j = 0; j < V2.size(); j++) {
			Mat[i][j] = V1[i] * V2[j];
		}
	}
	return Mat;
}

vector < vector<float>> OpenMPANN::transpose(vector<vector<float> >& m) {  //matrix transpose
	vector<vector<float>> matrixTrans(m[0].size(), vector<float>());

	for (int i = 0; i < m.size(); i++)
	{
		for (int j = 0; j < m[i].size(); j++)
		{
			matrixTrans[j].push_back(m[i][j]);
		}
	}

	return matrixTrans;
}

vector < vector<float>> OpenMPANN::m_m(vector< vector<float> > A, vector< vector<float> > B)  //matrix multiplication
{
	float a = 0;
	vector< vector<float> > M;
	M.resize(A.size(), vector<float>(B[0].size()));
	for (int i = 0; i < A.size(); i++) {
		for (int x = 0; x < B[0].size(); x++) {
			for (int y = 0; y < B.size(); y++) {
				a = a + (A[i][y] * B[y][x]);

			}
			M[i][x] = a;
			a = 0;
		}
	}
	return M;
}

vector<float> OpenMPANN::m_v(vector< vector<float> > X, vector<float> Y)
{
	vector<float> V;
	V.resize(X.size());

	int i;
	int j;
	double sum;

#  pragma omp parallel for num_threads(threadNum) \
	reduction(+: sum) default(none) private(i,j) shared(X,Y,V)
	for (i = 0; i < X.size(); i++)
	{
		sum = 0;
		for (j = 0; j < X[0].size(); j++) {
			sum += (X[i][j]) * (Y[j]);
		}
		V[i] = sum;
	}

	return V;
}

vector<int>  OpenMPANN::feedForward(vector< vector<float> > X) {
	vector<float> outPut;
	outPut.resize(10);
	predict.resize(X.size());
	feedForOut.resize(X.size(), vector<float>(10));

	int totalData;
	int i;
	int j;
	float pre;

//#  pragma omp parallel for num_threads(threadNum) \
   default(shared) private(totalData,i,j,pre,outPut)

	for (totalData = 0; totalData < X.size(); totalData++) {
		hiddenLayer_F[totalData][0] = m_v(firstWeight, X[totalData]);
		for (i = 0; i < hiddenLayer_F[totalData][0].size(); i++) {
			hiddenLayer_F[totalData][0][i] = (sigmoid(hiddenLayer_F[totalData][0][i] + firstBias[i]));
		}

		//testing(hiddenLayer);

		// hidden layer[1] to the last...
		for (i = 1; i < layerNum; i++) {
			hiddenLayer_F[totalData][i] = m_v(otherWeight[(i - 1)], hiddenLayer_F[totalData][i - 1]);
			for (j = 0; j < nodeNum; j++) {
				hiddenLayer_F[totalData][i][j] = (sigmoid(hiddenLayer_F[totalData][i][j] + otherBias[i - 1][j]));
			}
			//testing(otherWeight[i - 1]);
		}


		//Generate the output and predict

		outPut = m_v(lastWeight, hiddenLayer_F[totalData][(layerNum - 1)]);

		predict[totalData] = 0;
		pre = sigmoid(outPut[0] + lastBias[0]);

		for (i = 0; i < outPut.size(); i++) {

			outPut[i] = sigmoid(outPut[i] + lastBias[i]);

			// store last layer output for backpropagation
			feedForOut[totalData][i] = outPut[i];

			//prediction in Feed Forward
			if (outPut[i] > pre) {
				pre = outPut[i];
				predict[totalData] = i;
			}
		}

	}
	return predict;
}

void OpenMPANN::train(vector< vector<float> > X, vector<float> y) {
	vector<vector<float>> fw;

	//Mini-batch SGD algorithm
	for (int t = 0; t < stopEpochs; t++) {
		vector<int> preDict;

		cout << endl;
		cout << "Epochs: " << t + 1 << endl;
		cout << "-----------------------------------------------------------------------------------------" << endl;
		cout << "-----------------------------------------------------------------------------------------" << endl;

		//random shuffle and divide it into mini-batches d1,d2,d3,...,ds
		//suffle by time
		//unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		//shuffle(X.begin(), X.end(), default_random_engine(seed));
		//shuffle(y.begin(), y.end(), default_random_engine(seed));

		//suffle from 1 to 10
		int ran = (rand() % 10) + 1;
		shuffle(X.begin(), X.end(), default_random_engine(ran));
		shuffle(y.begin(), y.end(), default_random_engine(ran));

		dataSet.resize((X.size() / batchSize), vector <vector<float>>(batchSize, vector<float>(X[0].size())));
		ansSet.resize((X.size() / batchSize), vector<float>(batchSize));

		for (int i = 0; i < (X.size() / batchSize); i++) {
			for (int j = 0; j < batchSize; j++) {
				for (int k = 0; k < X[0].size(); k++) {
					dataSet[i][j][k] = X[(j + (i * batchSize))][k];
				}
				ansSet[i][j] = y[j + (i * batchSize)];
			}
		}

		for (int set = 0; set < (X.size() / batchSize); set++) {
			//prediction and calculate the accuracy
			fw.resize(nodeNum, vector<float>(dataSet[set][0].size()));
			cout << endl;
			cout << "Dataset:" << set + 1 << endl;
			cout << "-------------------------------------------------------------" << endl;
			cout << "-------------------------------------------------------------" << endl;

			preDict = feedForward(dataSet[set]);

			int correct = 0;
			for (int i = 0; i < preDict.size(); i++) {
				if (preDict[i] == ansSet[set][i]) {
					correct++;
				}
			}

			double Acuuracy = (double)correct / preDict.size();
			cout << "Acuuracy : " << Acuuracy * 100 << "%" << endl;

			vector<float> averageError;
			float Ferror;
			averageError.resize(10);

			for (int i = 0; i < batchSize; i++) {
				//cout << "batch= " << i << endl << endl;
				for (int j = 0; j < 10; j++) {
					//cout << "j= " << j << endl;
					if (j == ansSet[set][i])
					{
						averageError[j] += (0.5 * pow((1 - feedForOut[i][j]), 2));
						//cout << "1 error:" << feedForOut[i][j] << endl;
					}
					else {
						averageError[j] += (0.5 * pow((0 - feedForOut[i][j]), 2));
						//cout << "0 error:"<<feedForOut[i][j] << endl;
					}
				}
			}

			// display average error in batched dataset
			for (int i = 0; i < 10; i++) {

				averageError[i] = averageError[i] / 32;

				cout << "average error of" << i << " " << averageError[i] << endl;
			}

			//find out error of the other layer 
			for (int bSize = 0; bSize < batchSize; bSize++) {

				errorOfHiddenLayer.resize((layerNum - 1), vector<float>(nodeNum));

				//calculating between last hidden layer to output
				vector<float> DeltaWeiOfOut;
				vector<vector<float>> DWO;
				vector<float> FirstHiddenError; // E
				vector<vector<float>> HiddenError; // E
				for (int i = 0; i < feedForOut[bSize].size(); i++) {
					feedForOut[bSize][i] = (dsigmoid(feedForOut[bSize][i]));
				}
				DeltaWeiOfOut = v_v(averageError, feedForOut[bSize]);
				DeltaWeiOfOut = scalarV(DeltaWeiOfOut, learningRate);
				DWO = v_v_m(DeltaWeiOfOut, hiddenLayer_F[bSize][layerNum - 1]);

				//change the weight of last hidden layer to output
				for (int i = 0; i < 10; i++) {
					for (int j = 0; j < nodeNum; j++) {
						lastWeight[i][j] = lastWeight[i][j] - DWO[i][j];
					}
				}

				vector<float> DeltaH;
				vector<vector<float>> DeltaH_M;
				HiddenError.resize((layerNum - 1), vector<float>(nodeNum));

				FirstHiddenError = m_v(transpose(lastWeight), averageError);
				if (layerNum > 1) {
					for (int i = 0; i < hiddenLayer_F[bSize].size(); i++) {
						for (int n = 0; n < hiddenLayer_F[bSize][i].size(); n++) {
							hiddenLayer_F[bSize][i][n] = (dsigmoid(hiddenLayer_F[bSize][i][n]));
						}
					}
					for (int i = 1; i < layerNum; i++) {
						if (i == 1) {
							HiddenError[i - 1] = m_v(transpose(otherWeight[layerNum - i - 1]), FirstHiddenError);
							DeltaH = v_v(FirstHiddenError, hiddenLayer_F[bSize][i]);
							DeltaH = scalarV(DeltaH, learningRate);
							DeltaH_M = v_v_m(DeltaH, firstWeight[bSize]);
							for (int a = 0; a < nodeNum; a++) {
								for (int b = 0; b < firstWeight[0].size(); b++) {
									firstWeight[a][b] = firstWeight[a][b] - DeltaH_M[a][b];
								}
							}
						}
						else {
							HiddenError[i - 1] = m_v(transpose(otherWeight[layerNum - i - 1]), hiddenLayer_F[bSize][layerNum - i]);
							DeltaH = v_v(HiddenError[i - 1], hiddenLayer_F[bSize][i - 1]);
							DeltaH = scalarV(DeltaH, learningRate);
							DeltaH_M = v_v_m(DeltaH, hiddenLayer_F[bSize][layerNum - 1 - i]);
							for (int a = 0; a < nodeNum; a++) {
								for (int b = 0; b < otherWeight[layerNum - i - 1].size(); b++) {
									otherWeight[layerNum - i - 1][a][b] = otherWeight[layerNum - 1 - i][a][b] - DeltaH_M[a][b];
								}
							}
						}

					}
				}
				else { //change the weight between first input to first hidden layer
					vector<float> DeltaIn;
					vector<vector<float>> DeltaIn_M;
					vector<float> DWIBias;
					FirstHiddenError.resize(10);
					FirstHiddenError = m_v(transpose(lastWeight), averageError);
					for (int i = 0; i < hiddenLayer_F[bSize][0].size(); i++) {		//disigmoid the whole
						hiddenLayer_F[bSize][0][i] = (dsigmoid(hiddenLayer_F[bSize][0][i]));
					}

					DeltaIn = v_v(FirstHiddenError, feedForOut[bSize]);
					DeltaIn = scalarV(DeltaIn, learningRate);
					DeltaIn_M = v_v_m(DeltaIn, firstWeight[bSize]);


					//cout << firstWeight.size() << firstWeight[0].size() << DeltaIn_M.size() << DeltaIn_M[0].size()<< endl;
					for (int i = 0; i < nodeNum; i++) {
						for (int j = 0; j < firstWeight[0].size(); j++) {
							//cout << firstWeight.size() << firstWeight[0].size() << DeltaIn_M.size() << DeltaIn_M[0].size() << endl;
							firstWeight[i][j] = firstWeight[i][j] + DeltaIn_M[i][j];
						}
						//cout << i << endl;
					}
				}
			}

		}
	}
}

void OpenMPANN::inference(vector<float> X) {
	//predict corresponding digit
	vector<float> outPut;
	outPut.resize(10);

	hiddenLayer[0] = m_v(firstWeight, X);
	for (int i = 0; i < hiddenLayer[0].size(); i++) {
		hiddenLayer[0][i] = (sigmoid(hiddenLayer[0][i] + firstBias[i]));
	}

	//hidden layer[1] to the last...
	for (int i = 1; i < layerNum; i++) {
		hiddenLayer[i] = m_v(otherWeight[(i - 1)], hiddenLayer[i - 1]);
		for (int j = 0; j < nodeNum; j++) {
			hiddenLayer[i][j] = (sigmoid(hiddenLayer[i][j] + otherBias[i - 1][j]));
		}
	}

	outPut = m_v(lastWeight, hiddenLayer[(layerNum - 1)]);

	int preNum = 0;
	float preValue = 0;

	for (int i = 0; i < outPut.size(); i++) {
		outPut[i] = (sigmoid(outPut[i] + lastBias[i]));

		if (outPut[i] > preValue) {
			preValue = outPut[i];
			preNum = i;
		}
	}
	cout << "prediction : " << preNum;
}

void OpenMPANN::storeWeight(string fileName) {
	ofstream outFile;
	outFile.open(fileName);

	for (int i = 0; i < firstWeight.size(); i++)
	{
		for (int j = 0; j < firstWeight[0].size(); j++)
		{
			outFile << firstWeight[i][j] << endl;
		}
	}

	for (int i = 0; i < otherWeight.size(); i++)
	{
		for (int j = 0; j < otherWeight[0].size(); j++)
		{
			for (int k = 0; k < otherWeight[0][0].size(); k++)
			{
				outFile << otherWeight[i][j][k] << endl;
			}
		}
	}

	for (int i = 0; i < lastWeight.size(); i++)
	{
		for (int j = 0; j < lastWeight[0].size(); j++)
		{
			outFile << lastWeight[i][j] << endl;
		}
	}

	outFile.close();
	cout << "weight stored at file: " << fileName << endl;

}

void OpenMPANN::loadWeight(string fileName) {
	ifstream inFile;
	inFile.open(fileName);

	for (int i = 0; i < firstWeight.size(); i++)
	{
		for (int j = 0; j < firstWeight[0].size(); j++)
		{
			inFile >> firstWeight[i][j];
		}
	}

	for (int i = 0; i < otherWeight.size(); i++)
	{
		for (int j = 0; j < otherWeight[0].size(); j++)
		{
			for (int k = 0; k < otherWeight[0][0].size(); k++)
			{
				inFile >> otherWeight[i][j][k];
			}
		}
	}

	for (int i = 0; i < lastWeight.size(); i++)
	{
		for (int j = 0; j < lastWeight[0].size(); j++)
		{
			inFile >> lastWeight[i][j];
		}
	}

	cout << "weight loaded from file: " << fileName << endl;

}

#endif
