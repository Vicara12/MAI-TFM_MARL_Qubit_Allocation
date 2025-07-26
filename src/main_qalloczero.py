from utils.timer import Timer
from random import random

t = Timer("t1")

@t.timer_decorator
def sum_rand(times):
  total = 0
  for _ in range(times):
    total += random()
  return total

def main():
  s=sum_rand(1000)
  t = Timer.get("t1")
  print(f"t = {t.time}, f={t.freq}, s={s}")


if __name__ == "__main__":
  main()