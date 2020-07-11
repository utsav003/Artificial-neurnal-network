import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer() 
import numpy as np
import time
import datetime

import json
import csv
from data import *
import speech_recognition as sr
import pyaudio
import wave

class first:
    def __init__(self):
        print ("%s sentences in training data" % len(training_data))

        self.words = []
        self.classes = []
        documents = []
        ignore_words = ['?']
        # loop through each sentence in our training data
        for pattern in training_data:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern['sentence'])
            # add to our words list
            self.words.extend(w)
            # add to documents in our corpus
            documents.append((w, pattern['class']))
            # add to our classes list
            if pattern['class'] not in self.classes:
                self.classes.append(pattern['class'])

        # stem and lower each word and remove duplicates
        self.words = [stemmer.stem(w.lower()) for w in self.words if w not in ignore_words]
        self.words = list(set(self.words))

        # remove duplicates
        self.classes = list(set(self.classes))

        # print (len(documents), "documents")
        # print (len(self.classes), "classes", self.classes)
        # print (len(self.words), "unique stemmed words", self.words)


        training = []
        output = []
        # create an empty array for our output
        output_empty = [0] * len(self.classes)

        # training set, bag of words for each sentence
        for doc in documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
            # create our bag of words array
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            training.append(bag)
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            output.append(output_row)

        # sample training/output
        i = 0
        w = documents[i][0]
        print ([stemmer.stem(word.lower()) for word in w])
        print (training[i])
        print (output[i])



        self.ERROR_THRESHOLD = 0.1
        # load our calculated synapse values
        synapse_file = 'artificial_neural_network.json' 
        with open(synapse_file) as data_file: 
            synapse = json.load(data_file) 
            self.synapse_0 = np.asarray(synapse['synapse0']) 
            self.synapse_1 = np.asarray(synapse['synapse1'])


    def sigmoid(self,x):
        output = 1/(1+np.exp(-x))
        return output

    # convert output of sigmoid function to its derivative
    def sigmoid_output_to_derivative(self,output):
        return output*(1-output)


    def clean_up_sentence(self,sentence):
        # tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self, sentence, words, show_details=False):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words
        bag = [0]*len(self.words)  
        for s in sentence_words:
            for i,w in enumerate(self.words):
                if w == s: 
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)

        return(np.array(bag))

    def think(self, sentence, show_details=False):
        x = self.bow(sentence.lower(), self.words, show_details)
        if show_details:
            print ("sentence:", sentence, "\n bow:", x)
        # input layer is our bag of words
        l0 = x
        # matrix multiplication of input and hidden layer
        l1 = self.sigmoid(np.dot(l0, self.synapse_0))
        # output layer
        l2 = self.sigmoid(np.dot(l1, self.synapse_1))
        return l2




    def classify(self, sentence, show_details=False):
        results = self.think(sentence, show_details)
        data = {}
        didumean = []
        results = [[i,r] for i,r in enumerate(results) if r>=self.ERROR_THRESHOLD ] 
        # print(results[0][1])
        try:
            res = results[0][1]
            a = res * 100
            print(a)
            if(a >= 90):
                results.sort(key=lambda x: x[1], reverse=True) 
                return_results =[[self.classes[r[0]],r[1]] for r in results]
                print ("%s \n classification: %s" % (sentence, return_results))
                # print("=================================>",return_results[0][0])
                arg = return_results[0][0]
                if(arg == "69:Switch User"):
                    l = sentence.split()
                    size = len(l)
                    for i in l:
                        if(i == 'user' or i == 'profile'):
                            before_keyword, keyword, after_keyword = sentence.partition(i)
                            print(after_keyword)
                            data['argument'] = after_keyword.lstrip()
                            data['voicecommand'] = return_results[0][0].split(':')[0]
                            data['commandname'] = "Switch user"

                elif(arg == "68:Connect to Wi-Fi"):
                    l = sentence.split()
                    size = len(l)
                    for i in l:
                        if(i == 'to' or i == 'wi-fi' or i == 'wifi' or i == 'wi fi' or i == "join" ):
                            before_keyword, keyword, after_keyword = sentence.partition(i)
                            # print(after_keyword)
                            data['argument'] = after_keyword.lstrip()
                            data['voicecommand'] = return_results[0][0].split(':')[0]
                            data['commandname'] = "Connect to Wi-Fi"

                elif(arg == '56:Search URL'):
                    l = sentence.split()
                    size = len(l)
                    for i in l:
                        if(i == 'url' or i == 'link' or i == 'search'):
                            before_keyword, keyword, after_keyword = sentence.partition(i)
                            # print(after_keyword)
                            data['argument'] = after_keyword.lstrip()
                            data['voicecommand'] = return_results[0][0].split(':')[0]
                            data['commandname'] = "Search URL"


                else:

                    data['voicecommand'] = return_results[0][0].split(':')[0]
                    data['commandname'] = return_results[0][0].split(':')[1]
                    data['argument'] = " "


            elif(a <= 90 and a >=10):
                results.sort(key=lambda x: x[1], reverse=True) 
                return_results =[[self.classes[r[0]],r[1]] for r in results]
                print ("%s \n classification: %s" % (sentence, return_results))
                # print(return_results)
                length = len(return_results)
                print(length)

                for i in range(length):
                    data1 = {}
                    print("did you mean")
                    print(return_results[i][0])
                    arg = return_results[i][0]
                    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",arg)
                    data1['argument'] = " "
                    if arg == "69:Switch User":
                        l = sentence.split()
                        size = len(l)
                        for j in l:
                            if(j == 'user' or j == "to" or j == 'profile'):
                                before_keyword, keyword, after_keyword = sentence.partition(j)
                                # print("!!!!!!!!!!!!!!!!!",after_keyword.lstrip())
                                data1['argument'] = str(after_keyword.lstrip())
                                data1['voicecommand'] = return_results[i][0].split(':')[0]
                                data1['commandname'] = "Switch user"
                                # print("!!!!!!!!!!!!!!!!!",data1)

                    elif arg == "68:Connect to Wi-Fi":
                        l = sentence.split()
                        size = len(l)
                        for k in l:
                            if(k == 'wi-fi' or k == 'wifi' or k == 'wi fi' or k == "join" or k == 'to'):
                                before_keyword, keyword, after_keyword = sentence.partition(k)
                                # print(after_keyword)
                                data1['argument'] = str(after_keyword.lstrip())
                                data1['voicecommand'] = return_results[i][0].split(':')[0]
                                data1['commandname'] = "Connect to Wi-Fi"

                    elif arg == "56:Search URL":
                        l = sentence.split()
                        size = len(l)
                        for k in l:
                            if(k == 'url' or k == 'link' or k == 'search'):
                                before_keyword, keyword, after_keyword = sentence.partition(k)
                                # print(after_keyword)
                                data1['argument'] = str(after_keyword.lstrip())
                                data1['voicecommand'] = return_results[i][0].split(':')[0]
                                data1['commandname'] = "Search URL"
                    else :

                        # print("!!!!!!!!!!!!!!!!!")
                        data1['voicecommand'] = return_results[i][0].split(':')[0]
                        data1['commandname'] = return_results[i][0].split(':')[1]
                        # print("--------------------",data1)
                    #data1['voicecommand'] = return_results[i][0]#.split(':')[0]
                    didumean.append(data1)#return_results[i][0].split(':')[0])

                data['Did you mean'] = didumean




            return data
        except Exception as e:
                    print("========================================================>No command found") 






# ress = p.classify("replay the video")
# ress = p.classify("switch user cerberoz")
# ress = p.classify("wifi on switch to tv")
# ress = p.classify("switch to tv")
# ress = p.classify("learning list")
n=0
while True:
	n = n +1
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 8000
	RECORD_SECONDS = 4
	ress = "output.wav"

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
	                channels=CHANNELS,
	                rate=RATE,
	                input=True,
	                frames_per_buffer=CHUNK)

	print("* recording")

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	    data = stream.read(CHUNK)
	    frames.append(data)

	print("* done recording")


	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(ress, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()  


	r = sr.Recognizer()

	hellow=sr.AudioFile('output.wav')
	with hellow as source:
	    audio = r.record(source)
	try:
	    inp = r.recognize_google(audio,language="en-IN")
	    ans = inp.lower()
	    print(ans)
	except Exception as e:
            print("No input given "+str(e))
	p = first()
	ress = p.classify(ans)
	print("##################",ress)

# p.classify("academics")
# p.classify("gk")
# p.classify("revision")
# p.classify("progress applciation")
# p.classify("tips")
# p.classify("test")
# p.classify("library")
# p.classify("edueye")
# p.classify("update application")
# p.classify("miracast")
# p.classify("browser")
# p.classify("faq")
# p.classify("wifi")
# p.classify("ethernet")
# p.classify("user profile")
# p.classify("display")
# p.classify("date")
# p.classify("usb")
# p.classify("learning list")
# p.classify("reset")
# p.classify("update setting")
# p.classify("about")
# p.classify("snakes")
# ress = p.classify("close")
# ress = p.classify("play")
# ress = p.classify("end")
# ress =  p.classify("exit")
# p.classify("play video")
# p.classify("pause video")
# p.classify("stop video")
# p.classify("next video")
# p.classify("previous video")
# p.classify("change audio")
# p.classify("mute")
# p.classify("unmute")
# p.classify("switch to video")
# p.classify("tv shows")
# p.classify("back")
# p.classify("forward url")
# p.classify("back url")
# p.classify("refresh url")
# p.classify("search url")
# p.classify("globe on")
# p.classify("globe off")
# p.classify("wi-fi enable")
# p.classify("wi-fi disable")
# p.classify("hotspot enable")
# p.classify("hotspot disable")
# p.classify("connect to ssid")
# p.classify("switch user")
# p.classify("logout")
# p.classify("restart")
# p.classify("shutdown")
# p.classify("move to apps")
# p.classify("open settings")
# p.classify("open trending")
# p.classify("open puzzle")
# p.classify("open hotstar")
# p.classify("open mxplayer")
# p.classify("replay video")

