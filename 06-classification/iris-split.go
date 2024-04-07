package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"

	"github.com/go-gota/gota/dataframe"
)

func getDataframe(file string) (dataframe.DataFrame, error) {
	options := dataframe.HasHeader(false)

	var df dataframe.DataFrame
	f, err := os.Open(file)
	if err != nil {
		return df, err
	}
	defer f.Close()

	return dataframe.ReadCSV(f, options), nil
}

func splitData() {
	df, err := getDataframe("../data/iris-copy.csv")
	if err != nil {
		fmt.Println(err)
		return
	}

	trainingNum := (4 * df.Nrow()) / 5

	// Shuffling data
	allIdx := make([]int, df.Nrow())
	for i := range allIdx {
		allIdx[i] = i
	}
	rand.Shuffle(len(allIdx), func(i, j int) { allIdx[i], allIdx[j] = allIdx[j], allIdx[i] })

	// Gunakan indeks acak untuk split
	trainingIdx := allIdx[:trainingNum]
	testIdx := allIdx[trainingNum:]

	trainingDF := df.Subset(trainingIdx)
	testDF := df.Subset(testIdx)

	setMap := map[int]dataframe.DataFrame{
		0: trainingDF,
		1: testDF,
	}

	for idx, setName := range []string{"iris-training.csv", "iris-test.csv"} {
		f, err := os.Create(setName)
		if err != nil {
			log.Fatal(err)
		}

		w := bufio.NewWriter(f)
		opt := dataframe.WriteHeader(false)
		if err := setMap[idx].WriteCSV(w, opt); err != nil {
			log.Fatal(err)
		}
	}
}

func main() {
	splitData()
}
