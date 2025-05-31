from tests.plottests import plottingDemo
from tests.traintests import trainDemo, basicTrainParams
from utils.modelutils import getTrainFolderPath, getFolderName


if __name__ == '__main__':
  # plottingDemo()
  # trainDemo(**basicTrainParams())
  print(getTrainFolderPath())
  print(getFolderName(10, "RandomSampler"))