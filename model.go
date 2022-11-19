package catboost

/*
#cgo linux CFLAGS: -Iheaders
#cgo darwin CFLAGS: -Iheaders

#cgo linux LDFLAGS: -L. -lcatboostmodel
#cgo darwin LDFLAGS: -L. -lcatboostmodel

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <model_calcer_wrapper.h>

static char** makeCharArray(int size)
{
        return calloc(sizeof(char*), size);
}

static void setArrayString(char **a, char *s, int n)
{
        a[n] = s;
}

static void freeCharArray(char **a, int size)
{
        int i;
        for (i = 0; i < size; i++)
                free(a[i]);
        free(a);
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func getError() error {
	messageC := C.GetErrorString()
	message := C.GoString(messageC)
	return fmt.Errorf(message)
}

func makeCStringArrayPointer(array []string) **C.char {
	cargs := C.makeCharArray(C.int(len(array)))
	for i, s := range array {
		C.setArrayString(cargs, C.CString(s), C.int(i))
	}
	return cargs
}

// Model is a wrapper over ModelCalcerHandler
type Model struct {
	Handler unsafe.Pointer
}

// GetFloatFeaturesCount returns a number of float features used for training
func (model *Model) GetFloatFeaturesCount() int {
	return int(C.GetFloatFeaturesCount(model.Handler))
}

// GetCatFeaturesCount returns a number of categorical features used for training
func (model *Model) GetCatFeaturesCount() int {
	return int(C.GetCatFeaturesCount(model.Handler))
}

// Close deletes model handler
func (model *Model) Close() {
	C.ModelCalcerDelete(model.Handler)
}

// LoadFullModelFromFile loads model from file
func LoadFullModelFromFile(filename string) (*Model, error) {
	model := &Model{}
	model.Handler = C.ModelCalcerCreate()
	if !C.LoadFullModelFromFile(model.Handler, C.CString(filename)) {
		return nil, getError()
	}
	return model, nil
}

// CalcModelPrediction returns raw predictions for specified data points
func (model *Model) CalcModelPrediction(floats [][]float32, floatLength int, cats [][]string, catLength int, numClasses int) ([]float64, error) {
	nSamples := len(floats)
	results := make([]float64, 5)

	floatsC := make([]*C.float, nSamples)
	for i, v := range floats {
		floatsC[i] = (*C.float)(C.calloc(C.sizeof_float, C.size_t(len(v))))
		C.memcpy(unsafe.Pointer(floatsC[i]), unsafe.Pointer(&v[0]), C.size_t(len(v))*C.sizeof_float)
		defer C.free(unsafe.Pointer(floatsC[i]))
	}

	catsC := make([]**C.char, nSamples)
	for i, v := range cats {
		pointer := makeCStringArrayPointer(v)
		defer C.freeCharArray(pointer, C.int(len(v)))
		catsC[i] = pointer
	}

	if !C.CalcModelPrediction(
		model.Handler,
		C.size_t(nSamples),
		(**C.float)(&floatsC[0]), C.size_t(floatLength),
		(***C.char)(&catsC[0]), C.size_t(catLength),
		(*C.double)(&results[]), C.size_t(nSamples),
	) {
		return nil, getError()
	}

	return results, nil
}
