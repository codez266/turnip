# turnip

The Turnip Triple Scorer for entity-relation ranking
This project is under active development for my undergraduate thesis.
See [WSDM cup](http://www.wsdm-cup-2017.org/triple-scoring.html) for more details.

To run the cnnrank for classifying single person profession pair:

	python cnnrank.py -i data/ -o output/ -pro

where **-i** specifies the directory for input data.
Currently train files are hardcoded. The format for train file is:

`<PersonName><Tab><ProfessionName><Tab><TextLength><Tab><TrainText>`

The last column could be further tab separated for different text sources.

##Current approaches:

* cnnrank.py is for experiments with CNNs and LSTMs and exploring their usability

For a comprehensive overview of the project head over to the [blog post](blog.sumitasthana.net/ug-research-project) as well as the [poster](blog.sumitasthana.net/resources/ug_thesis_poster.pdf)
