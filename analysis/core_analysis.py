#analysis\core_analysis.py
import pypylon
from pypylon import pylon
import cv2
import sys
import numpy as np
import pandas as pd
import serial
import threading
import time
import os 
from datetime import datetime
import random
import pygame 
from dataclasses import dataclass 
from typing import Callable, Optional 
from PyQt6.QtCore import QObject, pyqtSignal


def run_analysis(): 
    return None