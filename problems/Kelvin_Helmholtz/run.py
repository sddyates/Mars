import sys
sys.path.insert(0, '../../src')

from main import Main
from user import User

me = User()
Main(me)
