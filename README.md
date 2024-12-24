# FastText Golang Wrapper

## Overview

Here's my attempt at wrapping FastText C++ library with Golang CGO.

## Requirements

- `fasttext` library installed through `brew` or any other package manager in `/usr/local/lib`

## Compiling

- Build the Go package normally (in the `fasttext-go-wrapper/` dir)

    ```Bash
    go build
    ```

## Basic usage
- Initialization
    ```
    model, err = fasttext.New(modelName)
    if err != nil {
        panic(err)
    }
    ```
    
- Predict vector
    ```
    vec, err := model.GetSentenceVector(sentence)
    if err != nil {
        panic(err)
    }
    ```
be aware that this method returns a non-normalized vector

- Get model dimension
    ```
    d, err := model.GetDimension()
	if err != nil {
		panic(err)
	}
    ```
## Example of Dockerfile
    RUN apt-get update && apt-get install fasttext -y

    WORKDIR /src
    RUN go build .
