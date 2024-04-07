package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/knn"
	"gonum.org/v1/gonum/mat"
)

func getIrisModel(distanceType string, k int) (*knn.KNNClassifier, error) {
	var cls *knn.KNNClassifier
	trainData, err := base.ParseCSVToInstances("./iris-training.csv", false)
	if err != nil {
		return cls, err
	}

	cls = knn.NewKnnClassifier(distanceType, "linear", k)
	cls.Fit(trainData)

	return cls, nil
}

func main() {

	log := log.New(os.Stdout, "Go-ML : ", log.LstdFlags|log.Lmicroseconds|log.Lshortfile)
	irisModel, err := getIrisModel("euclidean", 2)
	if err != nil {
		log.Fatalf("error: create knn iris model: %s", err)
	}
	// parameter server
	server := http.Server{
		Addr:         "0.0.0.0:9111",
		Handler:      API(irisModel, log),
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
	}

	serverErrors := make(chan error, 1)
	// mulai listening server
	go func() {
		log.Println("server listening on", server.Addr)
		serverErrors <- server.ListenAndServe()
	}()

	// Membuat channel untuk mendengarkan sinyal interupsi/terminate dari OS.
	// Menggunakan channel buffered karena paket signal membutuhkannya.
	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, os.Interrupt, syscall.SIGTERM)

	// Mengontrol penerimaan data dari channel,
	// jika ada error saat listenAndServe server maupun ada sinyal shutdown yang diterima
	select {
	case err := <-serverErrors:
		log.Fatalf("error: listening and serving: %s", err)

	case <-shutdown:
		log.Println("caught signal, shutting down")

		// Jika ada shutdown, meminta tambahan waktu 5 detik untuk menyelesaikan proses yang sedang berjalan.
		const timeout = 5 * time.Second
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		if err := server.Shutdown(ctx); err != nil {
			log.Printf("error: gracefully shutting down server: %s", err)
			if err := server.Close(); err != nil {
				log.Printf("error: closing server: %s", err)
			}
		}
	}

	log.Println("done")
}

type app struct {
	mux *http.ServeMux
}

// API : implement a http.Handler interface
func API(knnIrisModel *knn.KNNClassifier, log *log.Logger) http.Handler {
	app := new(app)
	app.mux = http.NewServeMux()

	knnIris := KnnIris{Model: knnIrisModel, Log: log}

	app.mux.Handle("/", http.FileServer(http.Dir("./html")))

	app.mux.HandleFunc("/get-iris-class", func(w http.ResponseWriter, r *http.Request) {
		// Set CORS headers
		header := w.Header()
		header.Add("Access-Control-Allow-Origin", "*")
		header.Add("Access-Control-Allow-Methods", "DELETE, POST, GET, OPTIONS, PUT")
		header.Add("Access-Control-Allow-Headers", "Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With, Token")
		header.Add("Content-Type", "application/json; charset=utf-8")

		switch r.Method {
		case http.MethodGet:
			http.NotFound(w, r)
		case http.MethodPost:
			knnIris.GetClass(w, r)
		default:
			http.NotFound(w, r)
		}
	})

	return app
}

func (a *app) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	a.mux.ServeHTTP(w, r)
}

type Iris struct {
	SepalLength float64
	SepalWidth  float64
	PetalLength float64
	PetalWidth  float64
	Species     string
}

type KnnIris struct {
	Model *knn.KNNClassifier
	Log   *log.Logger
}

func (u *KnnIris) GetClass(w http.ResponseWriter, r *http.Request) {
	var list []Iris
	decoder := json.NewDecoder(r.Body)
	err := decoder.Decode(&list)
	if err != nil {
		u.Log.Printf("error decode list iris: %s", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	var ds []float64
	for _, v := range list {
		if v.SepalLength <= 0 || v.SepalWidth <= 0 || v.PetalLength <= 0 || v.PetalWidth <= 0 {
			u.Log.Printf("Bad Request for : %+v", v)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		ds = append(ds, v.SepalLength, v.SepalWidth, v.PetalLength, v.PetalWidth, 0.0)
	}

	mat := mat.NewDense(len(list), 5, ds)
	inst := base.InstancesFromMat64(len(list), 5, mat)
	attrs := inst.AllAttributes()
	inst.AddClassAttribute(attrs[4])

	predictions, err := u.Model.Predict(inst)
	if err != nil {
		u.Log.Printf("error predict iris: %s", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	for i := range list {
		if predictions.RowString(i) == "1.00" {
			list[i].Species = "Iris-setosa"
		} else if predictions.RowString(i) == "2.00" {
			list[i].Species = "Iris-versicolor"
		} else if predictions.RowString(i) == "3.00" {
			list[i].Species = "Iris-virginica"
		}
	}

	data, err := json.Marshal(list)
	if err != nil {
		u.Log.Println("error marshalling result", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(data); err != nil {
		log.Println("error writing result", err)
	}
}
