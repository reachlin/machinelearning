
with open('lottery.csv', 'r') as f:
  for line in f:
    print ','.join(line.split(',')[1:8])