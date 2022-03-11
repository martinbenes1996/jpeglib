
import numpy as np
import os
import shutil
import sys
import tempfile
import unittest

sys.path.append('.')
import jpeglib

class TestFlags(unittest.TestCase):
	def setUp(self):
		try: shutil.rmtree("tmp")
		except: pass
		finally: os.mkdir("tmp")
	def tearDown(self):
		shutil.rmtree("tmp")
	
	# def test_fancy_upsampling(self):
		
	# 	jpeglib.version.set('6b')
	# 	with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
	# 		print("default flags")
	# 		x_def = im.read_spatial(flags = [])
	# 		print("+DO_FANCY_UPSAMPLING")
	# 		x_fu = im.read_spatial(flags = ['+DO_FANCY_UPSAMPLING'])
	# 		print("-DO_FANCY_UPSAMPLING")
	# 		x_ss = im.read_spatial(flags = ['-DO_FANCY_UPSAMPLING'])
	# 	self.assertTrue((x_def == x_fu).all())
	# 	self.assertTrue((x_def != x_ss).any())

	def test_fancy_downsampling(self):
		jpeglib.version.set('9e')
		with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
			x = im.read_spatial(flags = ['-DO_FANCY_DOWNSAMPLING'])
		# default flags
		with tempfile.NamedTemporaryFile() as tmp:
			print("default DO_FANCY_DOWNSAMPLING")
			with jpeglib.JPEG() as im:
				im.write_spatial(tmp.name, x, flags = [])
			with jpeglib.JPEG(tmp.name) as im:
				Y_def, CbCr_def, qt_def = im.read_dct()
		# fancy upsampling
		with tempfile.NamedTemporaryFile() as tmp:
			print("+DO_FANCY_DOWNSAMPLING")
			with jpeglib.JPEG() as im:
				im.write_spatial(tmp.name, x, flags = ['+DO_FANCY_DOWNSAMPLING'])
			with jpeglib.JPEG(tmp.name) as im:
				Y_fu, CbCr_fu, qt_fu = im.read_dct()
		# simple scaling
		with tempfile.NamedTemporaryFile() as tmp:
			print("-DO_FANCY_DOWNSAMPLING")
			with jpeglib.JPEG() as im:
				im.write_spatial(tmp.name, x, flags = ['-DO_FANCY_DOWNSAMPLING'])
			with jpeglib.JPEG(tmp.name) as im:
				Y_ss, CbCr_ss, qt_ss = im.read_dct()
		
		np.testing.assert_array_equal(Y_def, Y_fu)
		np.testing.assert_array_equal(CbCr_def, CbCr_fu)
  
		self.assertFalse((Y_fu == Y_ss).all())
		self.assertTrue((CbCr_fu == CbCr_ss).all())

