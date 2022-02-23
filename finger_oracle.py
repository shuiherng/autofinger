#!/usr/bin/env python

class FingerOracle(object):
	# comfortable semitone distances for each consecutive finger pair
	# eg. from white-to-white key, comfortable semitone distance for thumb and first finger is -4 to 9 semitones.

	# white-to-white keys or black-to-black keys
	wwbb = [None, 
			 [None, (0, 0), (-4, 9), (-3, 11), (-1, 12), (0, 13)],
			 [None, (-7, 4), (0, 0), (0, 5), (0, 7), (3, 9)],
			 [None, (-10, 2), (-5, 0), (0, 0), (0, 3), (3, 7)],
			 [None, (-12, 2), (-7, 0), (-3, 0), (0, 0), (0, 4)],
			 [None, (-13, 0), (-12, 0), (-7, 0), (-3, 0), (0, 0)]]

	# white-to-black keys
	wb = [None,
		   [None, (0, 0), (-5, 9), (-4, 10), (-2, 10), (-1, 14)],
		   [None, (-9, 1), (0, 0), (-1, 4), (-1, 7), (3, 9)],
		   [None, (-11, 0), (-6, 0), (0, 0), (0, 4), (2, 6)],
		   [None, (-10, 0), (-8, 0), (-4, 0), (0, 0), (0, 3)],
		   [None, (-14, 4), (-9, 4), (-6, 1), (-5, 0), (0, 0)]]

	# black-to-white keys
	bw = [None,
		   [None, (0, 0), (-3, 8), (-2, 11), (-1, 13), (2, 13)],
		   [None, (-8, 4), (0, 0), (0, 4), (1, 8), (3, 10)],
		   [None, (-10, 3), (-5, 0), (0, 0), (0, 4), (3, 6)],
		   [None, (-10, 0), (-8, -1), (-3, -1), (0, 0), (0, 4)],
		   [None, (-13, 1), (-9, -2), (-6, -2), (-4, 0), (0, 0)]]

	def __init__(self, before, after, distance):
		# comparison array depends on which key types are played consecutively
		if before == after: # check if first and second key both white or black
			self.target = FingerOracle.wwbb
		elif before: # black key first, white key second
			self.target = FingerOracle.bw
		else: # white key first, black key second
			self.target = FingerOracle.wb

		self.distance = distance

	# returns range for type
	# i, j: fingers used to press k_before and k_after respectively
	# k_before, k_after: consecutive keys, boolean values representing whether it's a black key
	# distance: semitone distance between before and after.
	def is_valid(self, i, j):
		return self.target[i][j][0] <= self.distance <= self.target[i][j][1]