#!/usr/bin/env python3
import numpy as np
import dataProcessor, pickle, random
import argparse
import model_utils as mutil

def main():
    parser = argparse.ArgumentParser(description='LSTM arguments.')
    parser.add_argument('dtype', metavar='dtype', type=int, default=2,
                        help='data type: 0 - TRUMP, 1 - EN.DE, 2 - EN, 3 - TEST')
    parser.add_argument("--train", help="Train model.", action="store_true")
    parser.add_argument("--load", help="Train model.", action="store_true")
    parser.add_argument("--generate", help="Train model.", action="store_true")
    args = parser.parse_args()

    x_train, y_train, word_index, index_word, word_dim = dataProcessor.create_training_data(args.dtype)

    if args.train:
        mutil.train_lstm(word_dim=word_dim, x=x_train, y=y_train, should_load=args.load)

    if args.generate:
        pickled_LSTM = load_model()
        if args.dtype == dataProcessor.TrainingDataType.TEST:
            mutil.geneate_test(pickled_LSTM)
        else:
            mutil.generate_sentences(pickled_LSTM, word_index=word_index, index_word=index_word)

if __name__ == "__main__":
    main()