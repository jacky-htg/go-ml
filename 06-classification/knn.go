package main

import (
	"fmt"
	"math"

	"github.com/rocketlaunchr/dataframe-go"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
	"gonum.org/v1/gonum/mat"
)

func runFromMat() {
	trainData, err := base.ParseCSVToInstances("iris-training.csv", false)
	if err != nil {
		panic(err)
	}

	testData, err := base.ParseCSVToInstances("iris-test.csv", false)
	if err != nil {
		panic(err)
	}

	//Initialises a new KNN classifier
	cls := knn.NewKnnClassifier("euclidean", "linear", 2)

	// Jika baru punya rawData, bisa lakukan training-test split
	//trainData, testData := base.InstancesTrainTestSplit(rawData, 0.80)
	cls.Fit(trainData)

	fmt.Println(trainData)

	//Calculates the Euclidean distance and returns the most popular label
	predictions, err := cls.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Prints precision/recall metrics
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMat))

	// deploy new data
	mat := mat.NewDense(1, 5, []float64{
		5.1, 3.5, 1.4, 0.2, 0.0,
		//5.6, 2.7, 4.2, 1.3, 0.0,
	})
	inst := base.InstancesFromMat64(1, 5, mat)
	attrs := inst.AllAttributes()
	inst.AddClassAttribute(attrs[4])
	fmt.Println(inst)
	//Calculates the Euclidean distance and returns the most popular label
	predictions, err = cls.Predict(inst)
	if err != nil {
		panic(err)
	}
	fmt.Println(predictions)
}

func runFromDataframe() {
	rawData, err := base.ParseCSVToInstances("../data/iris.csv", true)
	if err != nil {
		panic(err)
	}

	//fmt.Println(rawData)
	//Initialises a new KNN classifier
	cls := knn.NewKnnClassifier("euclidean", "linear", 2)

	//Do a training-test split
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.80)
	cls.Fit(trainData)

	//Calculates the Euclidean distance and returns the most popular label
	predictions, err := cls.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Prints precision/recall metrics
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMat))

	s1 := dataframe.NewSeriesFloat64("Sepal Length", nil, 5.1, 7.0, 6.3)
	s2 := dataframe.NewSeriesFloat64("Sepal Width", nil, 3.5, 3.2, 3.3)
	s3 := dataframe.NewSeriesFloat64("Petal Length", nil, 1.4, 4.7, 6.0)
	s4 := dataframe.NewSeriesFloat64("Petal Width", nil, 0.2, 1.4, 2.5)
	s5 := dataframe.NewSeriesString("Species", nil, "Iris-setosa", "Iris-versicolor", "Iris-virginica")
	df := dataframe.NewDataFrame(s1, s2, s3, s4, s5)

	fmt.Println(df)

	newInstance := base.ConvertDataFrameToInstances(df, 4)
	fmt.Println(newInstance)

	//Calculates the Euclidean distance and returns the most popular label
	predictions, err = cls.Predict(newInstance)

	if err != nil {
		panic(err)
	}

	fmt.Println(predictions)
}

func generic() {
	// Read in the iris data set into golearn "instances".
	irisData, err := base.ParseCSVToInstances("../data/iris.csv", true)
	if err != nil {
		panic(err)
	}

	// Initialize a new KNN classifier.  We will use a simple
	// Euclidean distance measure and k=2.
	knn := knn.NewKnnClassifier("euclidean", "linear", 2)

	// Use cross-fold validation to successively train and evalute the model
	// on 5 folds of the data set.
	cv, err := evaluation.GenerateCrossFoldValidationConfusionMatrices(irisData, knn, 5)
	if err != nil {
		panic(err)
	}

	// Get the mean, variance and standard deviation of the accuracy for the
	// cross validation.
	mean, variance := evaluation.GetCrossValidatedMetric(cv, evaluation.GetAccuracy)
	stdev := math.Sqrt(variance)

	// Output the cross metrics to standard out.
	fmt.Printf("\nAccuracy\n%.2f (+/- %.2f)\n\n", mean, stdev*2)
}

func main() {
	//generic()
	//runFromDataframe()
	runFromMat()
}
