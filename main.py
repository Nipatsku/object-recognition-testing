from objectRecognition import object_recognition
from googleImageSearch import google_image_search_download
import random


def main():
    query = 'squirrel funny haha'
    query_length = 5
    class_name = 'fox_squirrel'
    probability_threshold = 50.0

    # download google image search results.
    images = google_image_search_download(query, query_length)

    # predict objects in images and
    # list images with expected class name + high enough probability
    positives = []
    for i, image_path in enumerate(images):
        print(f'Recognizing objects ... {i+1}/{len(images)}')
        print(f'\t{image_path}')
        try:
            predictions, probabilities = object_recognition(image_path, 5)
            for eachPrediction, eachProbability in zip(predictions, probabilities):
                print('\t', eachPrediction, " : ", eachProbability, "%")
                if eachProbability >= probability_threshold and eachPrediction == class_name:
                    positives.append(image_path)
                    break
        except Exception as e:
            print(f'\tUnexpected error: {e}')

    print(f'Identified {len(positives)} positives out of {len(images)} images!')
    print(positives)

    if len(positives) == 0:
        return None

    # select random result
    result = random.choice(positives)
    print(f'\n{result}')
    return result

if __name__ == '__main__':
    main()