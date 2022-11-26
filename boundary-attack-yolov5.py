# Import needed libraries
import numpy as np
import time
import os
from PIL import Image
import torch
import sys
import random

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


class BoundaryAttackYolov5():

	def __init__(	self,
					classifier=r'yolov5x.pt'	):

		self.RESNET_MEAN = np.array([103.939, 116.779, 123.68])
		self.current_directory = os.path.dirname(os.path.abspath(__file__))
		self.classifier = torch.hub.load(	os.path.join(self.current_directory, r'yolov5'),
											'custom',
											path=os.path.join(self.current_directory, r'models', classifier),
											source='local'	)
  
	def orthogonal_perturbation(self, delta, prev_sample, target_sample):
		"""Generate orthogonal perturbation."""
		perturb = np.random.randn(224, 224, 3)
		perturb /= np.linalg.norm(perturb, axis=(0, 1))
		perturb *= delta * np.mean(self.get_diff(target_sample, prev_sample))
		# Project perturbation onto sphere around target
		diff = (target_sample - prev_sample).astype(np.float32)  # Orthorgonal vector to sphere surface
		diff /= self.get_diff(target_sample, prev_sample)  # Orthogonal unit vector
		# We project onto the orthogonal then subtract from perturb
		# to get projection onto sphere surface
		perturb -= (np.vdot(perturb, diff) / np.linalg.norm(diff) ** 2) * diff
		# Check overflow and underflow
		overflow = (prev_sample + perturb) - 255 + self.RESNET_MEAN
		perturb -= overflow * (overflow > 0)
		underflow = -self.RESNET_MEAN
		perturb += underflow * (underflow > 0)
		return perturb

	def forward_perturbation(self, epsilon, prev_sample, target_sample):
		"""Generate forward perturbation."""
		perturb = (target_sample - prev_sample).astype(np.float32)
		perturb *= epsilon
		return perturb

	def save_image(self, sample, classifier, name_folder, n_calls):
		"""Export image file."""
		prediction = self.predict(sample, classifier, image_size=640)
		label, confidence = prediction['name'], prediction['confidence']
		# Reverse preprocessing, see https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
		sample += self.RESNET_MEAN
		sample = sample[..., ::-1].astype(np.uint8)
		# Convert array to image and save
		sample = Image.fromarray(sample)
		id_no = time.strftime('%Y%m%d_%H%M%S', time.localtime())
		# Save with predicted label for image (may not be adversarial due to uint8 conversion)
		sample.save(os.path.join(self.current_directory, r'images', name_folder,
								"{}_{}_{}_{}.png".format(id_no, label, confidence, n_calls)))

	def preprocess(self, sample_path):
		"""Load and preprocess image file."""
		img = image.load_img(sample_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = preprocess_input(x)
		return x

	def get_diff(self, sample_1, sample_2):
		"""Channel-wise norm of difference between samples."""
		return np.linalg.norm(sample_1 - sample_2, axis=(0, 1))

	def predict(self, sample, classifier, image_size=640):
		sample = (sample + self.RESNET_MEAN).astype(np.uint8).astype(np.float32) - self.RESNET_MEAN
		sample = (sample + self.RESNET_MEAN).astype(np.uint8).astype(np.float32)
		sample = sample[..., ::-1].astype(np.uint8)
		classes = classifier(sample, size=image_size).pandas().xyxy[0]
		keys = list(classes.to_dict('list').keys())
		top_class = classes.iloc[0, :].tolist() if not classes.empty else [None, None, None, None, 0, None, 'Noise']
		top_class = dict(zip(keys, top_class))
		return top_class

	# single_class 
	def boundary_attack(	self,
							adversarial_sample,
							target_sample,
							targated_attack=False,
							single_class=False,
							probability_inference=0.1,
							epsilon=1, 
							delta=0.1	):
		# Setting minimum probability to eliminate weak predictions
		self.classifier.conf = probability_inference
		path_target_sample = os.path.join(self.current_directory, r'images', r'original_', target_sample)
		path_adversarial_sample = os.path.join(self.current_directory, r'images', r'original_', adversarial_sample)
		initial_sample = self.preprocess(path_adversarial_sample)
		target_sample = self.preprocess(path_target_sample)

		if targated_attack:
			attack_class = self.predict(np.copy(initial_sample), self.classifier, image_size=640)['class']
			name_attack_class = self.predict(np.copy(initial_sample), self.classifier, image_size=640)['name']
		else:
			attack_class = None
			name_attack_class = 'Noise'
		target_class = self.predict(np.copy(target_sample), self.classifier, image_size=640)['class']
		name_target_class = self.predict(np.copy(target_sample), self.classifier, image_size=640)['name']
		
		if target_class == attack_class:
			raise Exception(f'the image in {path_target_sample} and the image in {path_adversarial_sample} must not contain the same main object ')
		if target_class == 'None':
			raise Exception(f'The image contained in {path_target_sample} does not contain any object to detect')
		if targated_attack and attack_class == None:
			raise Exception(f'The image contained in {path_adversarial_sample} must contain objects to detect".')
		if not targated_attack and attack_class != None:
			raise Exception(f'The image contained in {path_adversarial_sample} must not contain any objects to be detected')
			
		folder = 'targated_attack_' if targated_attack else 'non_targated_attack_'
		folder += name_target_class + '_vs_' + name_attack_class + '_' + \
					str(probability_inference) + '_' + \
					time.strftime('%Y%m%d_%H%M%S', time.localtime()) 
		
		try : 
			os.mkdir(os.path.join(self.current_directory, r"images", folder))
		except FileExistsError as e:
			print(e), sys.exit()
		self.save_image(np.copy(initial_sample), self.classifier, folder, n_calls=0)

		adversarial_sample = initial_sample
		n_steps = 0
		n_calls = 0
		epsilon = epsilon
		delta = delta

		# Move first step to the boundary
		while True:
			trial_sample = adversarial_sample + self.forward_perturbation(epsilon, adversarial_sample, target_sample)
			prediction = self.predict(np.copy(trial_sample), self.classifier, image_size=640)['class']
			n_calls += 1
			if prediction == attack_class:
				adversarial_sample = trial_sample
				break
			else:
				epsilon *= 0.9

		# Iteratively run attack
		while True:
			print("Step #{}...".format(n_steps))
			# Orthogonal step
			print("\tDelta step...")
			d_step = 0
			while True:
				d_step += 1
				print("\t#{}".format(d_step))
				trial_samples = []
				for i in np.arange(10):
					trial_sample = adversarial_sample + self.orthogonal_perturbation(delta, adversarial_sample, target_sample)
					trial_samples.append(trial_sample)
				if not single_class: 
					trial_samples.sort(key=lambda sample:self.predict(np.copy(sample), self.classifier, image_size=640)['confidence'], reverse=True)
				
				predictions = []
				for sample in trial_samples:
					predictions.append(self.predict(np.copy(sample), self.classifier, image_size=640)['class'])

				n_calls += 10
				if not targated_attack:
					d_score = np.mean(np.array(predictions) != target_class)
				else:
					d_score = np.mean(np.array(predictions) == attack_class)

				if d_score > 0.0:
					if d_score < 0.3:
						delta *= 0.9
					elif d_score > 0.7:
						delta /= 0.9
					# adversarial_sample = np.array(trial_samples)[random.choice(np.where(np.array(predictions) != target_class)[0].tolist())]
					if not targated_attack:
						adversarial_sample = np.array(trial_samples)[np.where(np.array(predictions) != target_class)[0][0]]
					else:
						adversarial_sample = np.array(trial_samples)[np.where(np.array(predictions) == attack_class)[0][0]]
					break
				else:
					delta *= 0.9
			# Forward step
			print("\tEpsilon step...")
			e_step = 0
			while True:
				e_step += 1
				print("\t#{}".format(e_step))
				trial_sample = adversarial_sample + self.forward_perturbation(epsilon, adversarial_sample, target_sample)
				prediction = self.predict(np.copy(trial_sample), self.classifier, image_size=640)['class']
				n_calls += 1
				if not targated_attack :
					if prediction != target_class:
						adversarial_sample = trial_sample
						epsilon /= 0.5
						break
					elif e_step > 500:
						break
					else:
						epsilon *= 0.5
				else:
					if prediction == attack_class:
						adversarial_sample = trial_sample
						epsilon /= 0.5
						break
					elif e_step > 500:
						break
					else:
						epsilon *= 0.5

			n_steps += 1
			chkpts = [1, 5, 10, 50, 100, 500]
			if (n_steps in chkpts) or (n_steps % 500 == 0):
				print("{} steps".format(n_steps))
				self.save_image(np.copy(adversarial_sample), self.classifier, folder, n_calls)
			diff = np.mean(self.get_diff(adversarial_sample, target_sample))
			if diff <= 1e-3 or e_step > 500:
				print("{} steps".format(n_steps))
				print("Mean Squared Error: {}".format(diff))
				self.save_image(np.copy(adversarial_sample), self.classifier, folder, n_calls)
				break

			print("Mean Squared Error: {}".format(diff))
			print("Calls: {}".format(n_calls))
			print("Attack Class: {}".format(attack_class))
			print("Target Class: {}".format(target_class))
			print("Adversarial Class: {}".format(prediction))

if __name__ == "__main__":
    # r'adv_img.jpg'
    # r'gaussian_noise.jpg'
    # r'angry-dog.jpg'
	# r'elephant.jpg'
	# r'cat.jpg'
	# r'dog.jpg'  # dog_.jpg  # dog__.jpg
	# r'ourson.jpg'
	# r'zebra.jpg'
	# r'giraffe_.jpg'
	# r'plate2.jpg'
	# r'voiture_ancienne.jpg'
	boundary_attack_yolov5 = BoundaryAttackYolov5(classifier=r'best.pt')
	boundary_attack_yolov5.boundary_attack(	adversarial_sample=r'adv_img.jpg',
											target_sample=r'voiture_ancienne.jpg',
											probability_inference=0.01,
											targated_attack=False,
											single_class=True,
											epsilon=1, 
											delta=0.1	)