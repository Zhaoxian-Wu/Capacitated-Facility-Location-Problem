import pickle
import time

import matplotlib.pyplot as plt

fileName = '.drawData'
record = None

def draw_GA_population():
  # record[0]['title'] = 'RX'
  # record[1]['title'] = 'BX'
  for r in record:
    # plt.plot(r['x-axis'], r['y-axis'], label=r['title'])
    # plt.plot(r['x-axis'], r['y-axis'], label=r['k_tournament'])
    # plt.plot(r['x-axis'], r['y-axis'], label=r['reproduceProp'])
    # plt.plot(r['x-axis'], r['y-axis'], label=r['inheritanceProp'])
    plt.plot(r['x-axis'], r['y-axis'], label=r['population'])

    # timeStamp = time.strftime('%m%d%H%M%S', time.localtime())
    plt.title('GA')
    # print(r['population'])
    # print(r['genaration'])
    # print(r['k_tournament'])
    # print(r['changeProp'])
  plt.legend()
  plt.show()

def draw_SA_population():
  for r in record:
    # plt.plot(r['x-axis'], r['y-axis'], label=r['T0'])
    plt.plot(r['x-axis'], r['y-axis'], 
      label='T0:{} factor:{} Te: {}'.format(
        r['T0'], r['factor'], r['STOP_TEMPERATURE']))
    # plt.plot(r['x-axis'], r['y-axis'], label=r['reproduceProp'])
    # plt.plot(r['x-axis'], r['y-axis'], label=r['inheritanceProp'])
    # plt.plot(r['x-axis'], r['y-axis'], label=r['population'])

    # timeStamp = time.strftime('%m%d%H%M%S', time.localtime())
    plt.title('GA')
    # print(r['population'])
    # print(r['genaration'])
    # print(r['k_tournament'])
    # print(r['changeProp'])
  plt.legend()
  plt.show()

def remove():
  record.pop(1)
  with open(fileName, 'wb') as f:
    pickle.dump(record, f)

def init():
  with open(fileName, 'wb') as f:
    pickle.dump([], f)

if __name__ == '__main__':
  with open(fileName, 'rb') as f:
    record = pickle.load(f)
  draw_GA_population()
  # draw_SA_population()
