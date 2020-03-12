#! /usr/bin/env python

'''
MichelStyle, made by Remideza for Bend the Future single "Otaniemi"
'''
import time
import cv2
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import PIL
import os
import argparse
import sys

from pathlib import Path

FPS = 24
MAXBATCH = 100
VIDEOCOMBINE = True
AUDIOCOMBINE = True

def main():
	todoi = []
	paramsname = {}
	reset = True
	lastindex = 0
	picbatch = []

	for filename in  Path('./params').iterdir():
		stem = filename.stem
		if stem.isdigit():
			n = int(stem)
			ncalc = 1 if n==0 else int((n/100) * FPS)
			paramsname[ncalc] = stem
			todoi.append(ncalc)

			# if n == 0:	
			# 	paramsname[1] = stem
			# 	n = 1
			# else :
			# 	ncalc = int((n/100) * FPS)
			# 	paramsname[ncalc] = str(n)
			# 	n = ncalc
			# todoi.append(n)
		else:
			print(f'Skipping non-integer file: {filename}')
			continue
		
			
	todoi.sort()	

	donefiles = list([f.stem for f in Path('./temp').iterdir()])

	# donefiles = []
	# for filename in os.listdir('./temp'):
	# 	donefiles.append(int(filename[:-4]))
	# ETJ DEBUG
	print(f'donefiles: {donefiles}')
	# END DEBUG
	framesdones = len(donefiles)

	maxfile = int(todoi[-1])
	todo = maxfile - framesdones
	total = todo

	content_image =  cv2.imread('cover.jpg', cv2.IMREAD_COLOR)
	content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
	hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

	estime = "unknown"
	infoscreen(todo, estime)
	starttime = time.time()
	for i in range(1, maxfile+1):
		if i in todoi:
			lastindex = todoi.index(i)
		if i not in donefiles:
			if i in todoi:
				style_image = cv2.imread('./params/' + paramsname[i] + '.jpg', cv2.IMREAD_COLOR)
				reset = True
			else:
				if reset:
					im1 = cv2.imread('./params/' + paramsname[todoi[lastindex]] + '.jpg', cv2.IMREAD_COLOR)
					im2 = cv2.imread('./params/' + paramsname[todoi[lastindex+1]] + '.jpg', cv2.IMREAD_COLOR)
					reset = False
					total = todoi[lastindex+1] - todoi[lastindex]
				idx = i - todoi[lastindex]
				pc = idx/total
				style_image = cv2.addWeighted(im1, 1-pc, im2, pc, 0.0)
			style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB) 
			style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
			outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
			stylized = tensor_to_image(outputs[0])
			picbatch.append([stylized, i])
			if(len(picbatch) >= MAXBATCH):
				for p in picbatch:
					p[0].save('./temp/' + str(p[1]) +".jpg")		
					
				picbatch = []
				
			
			estime = int(((time.time() - starttime) * todo)/60)
			starttime = time.time()
			todo -= 1
			infoscreen(todo+1, estime)
			
	for p in picbatch:
		p[0].save('./temp/' + str(p[1]) +".jpg")	

	print("Done generating frames")
	if not VIDEOCOMBINE:
		exit()

	HEIGHT, WIDTH, channels = cv2.imread('./temp/1.jpg', 1).shape
	video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*"mp4v"), FPS,(WIDTH * 2,HEIGHT * 2))	
	print("Compiling video ...")
	for i in range(1, maxfile+1):
		sys.stdout.flush()
		sys.stdout.write(str(i) + "/" + str(maxfile) + "\r")
		video.write(cv2.resize(cv2.imread('./temp/' + str(i) +'.jpg',1), (WIDTH * 2,HEIGHT * 2)))
	video.release()

	print("Done !")

	if AUDIOCOMBINE and os.path.isfile('./audio.mp3'):
		import moviepy.editor as mp
		video = mp.VideoFileClip('video.mp4', FPS)
		video.write_videofile('output.mp4',audio='audio.mp3')

def clear():
	if sys.platform == 'darwin':
		os.system('clear')
	else: 
		os.system('cls')

def tensor_to_image(tensor):
	tensor = tensor*255
	tensor = np.array(tensor, dtype=np.uint8)
	if np.ndim(tensor)>3:
		assert tensor.shape[0] == 1
		tensor = tensor[0]
	return PIL.Image.fromarray(tensor)

def infoscreen(todo, time):
	clear()
	print("--------------------------")
	print("  MichelStyle - Remideza  ")
	print("--------------------------")
	print()
	print(str(todo) + " remaining frames to work on")
	print(str(time) + " minutes remaining")

# ETJ DEBUG
def main2():
	# Get arguments from command line
	# Find all params files & put them in order
	# Find which files still need to be done
	# Apply style to each remaining file & save out
	# combine files into video
	# Add audio if supplied
	
	pass
# END DEBUG

if __name__ == '__main__':
	main()