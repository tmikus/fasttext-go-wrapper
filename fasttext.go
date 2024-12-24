package fasttext

// #cgo CXXFLAGS: -std=c++11
// #cgo LDFLAGS: -L/opt/homebrew/lib -L/usr/local/lib -lfasttext  -lm -pthread
// #include <stdlib.h>
// #include "fasttext-wrapper.hpp"
//
// int ft_load_model(const char *path);
// go_fast_text_pair_t* ft_predict(const char *query_in, int k, float threshold, int* result_length);
// int ft_get_vector_dimension();
// int ft_get_sentence_vector(const char* query_in, float* vector, int vector_size);
// int train(const char* model_name, const char* input, const char* output, int epoch, int word_ngrams, int thread, float lr);
// int quantize(const char* input, const char* output);
// int ft_save_model(const char* filename);
// int ft_delete();
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

const (
	_ = iota

	cArraySize = 1 << 28

	// LabelA is an example prediction value Label
	LabelA

	// LabelB is an example prediction value Label
	LabelB

	// NoLabel is an example prediction value Label
	NoLabel

	error_init_model = "the fasttext model needs to be initialized first. it's should be done inside the `New()` function"
)

// Model uses FastText for it's prediction
type Model struct {
	isInitialized bool
}

type Prediction struct {
	Label string
	Prob  float32
}

// New should be used to instantiate the model.
// FastTest needs some initialization for the model binary located on `file`.
func New(file string) (*Model, error) {

	status := C.ft_load_model(C.CString(file))

	if status != 0 {
		return nil, fmt.Errorf("cannot initialize model on `%s`", file)
	}

	return &Model{
		isInitialized: true,
	}, nil
}

func (m *Model) GetDimension() (int, error) {

	if !m.isInitialized {
		return -1, errors.New(error_init_model)
	}

	res := int(C.ft_get_vector_dimension())
	if res == -1 {
		return res, errors.New("model is not initialized")
	}
	return res, nil
}

func (m *Model) Predict(text string, k int, threshold float32) []Prediction {
	textChar := C.CString(text)
	defer C.free(unsafe.Pointer(textChar))

	var (
		cPredictionsLen C.int
		cPredictionsPtr *C.go_fast_text_pair_t
	)

	cPredictionsPtr = C.ft_predict(C.CString(text), C.int(k), C.float(threshold), &cPredictionsLen)
	predictionsLen := int(cPredictionsLen)

	cPredictionsArray := (*[cArraySize]C.go_fast_text_pair_t)(unsafe.Pointer(cPredictionsPtr))[:predictionsLen:predictionsLen]

	return cArrayToGoSlice(cPredictionsArray)
}

// GetSentenceVector the `keyword`
func (m *Model) GetSentenceVector(keyword string) ([]float64, error) {
	if !m.isInitialized {
		return nil, errors.New(error_init_model)
	}

	vecDim := C.ft_get_vector_dimension()
	if vecDim <= 0 {
		return nil, fmt.Errorf("the dimension of the model `%d`is srtictly less than 0", vecDim)
	}
	var cfloat C.float
	result := (*C.float)(C.malloc(C.ulong(vecDim) * C.ulong(unsafe.Sizeof(cfloat))))

	defer C.free(unsafe.Pointer(result))

	keywordC := C.CString(keyword)

	defer C.free(unsafe.Pointer(keywordC))

	status := C.ft_get_sentence_vector(
		keywordC,
		result,
		vecDim,
	)

	if status != 0 {
		return nil, fmt.Errorf("exception when predicting `%s`", keyword)
	}
	p2 := (*[1 << 30]C.float)(unsafe.Pointer(result))
	ret := make([]float64, int(vecDim))
	for i := 0; i < int(vecDim); i++ {
		ret[i] = float64(p2[i])
	}

	return ret, nil
}

func (m *Model) SaveModel(filename string) error {
	if !m.isInitialized {
		return errors.New(error_init_model)
	}

	status := C.ft_save_model(C.CString(filename))

	if status != 0 {
		return fmt.Errorf("error while loading fasttext model to a `%s`", filename)
	}
	return nil
}

func (m *Model) Delete() error {
	if !m.isInitialized {
		return errors.New(error_init_model)
	}
	status := C.ft_delete()
	if status != 0 {
		return fmt.Errorf("error while deleting fasttext model")
	}
	return nil
}

func Train(model_name, input, output string, epoch, word_ngrams, thread int, lr float64) error {

	status := C.train(C.CString(model_name), C.CString(input), C.CString(output), C.int(epoch), C.int(word_ngrams), C.int(thread), C.float(lr))

	if status != 0 {
		return fmt.Errorf("error while training `%s` fasttext model", model_name)
	}
	// m.isInitialized = true
	return nil
}

func Quantize(input, output string) error {

	status := C.quantize(C.CString(input), C.CString(output))

	if status != 0 {
		return fmt.Errorf("error while quantizing `%s` fasttext model", input)
	}
	return nil
}

func cArrayToGoSlice(cArray []C.go_fast_text_pair_t) []Prediction {
	predictions := make([]Prediction, 0, len(cArray))

	for _, cStruct := range cArray {
		prediction := Prediction{
			Label: C.GoString(cStruct.label),
			Prob:  float32(cStruct.prob),
		}
		predictions = append(predictions, prediction)
	}

	return predictions
}
