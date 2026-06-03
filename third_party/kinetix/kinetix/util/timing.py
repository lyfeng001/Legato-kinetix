from timeit import default_timer as tmr

counter = 0


def time_function(f, name):
    global counter
    t = "\t" * counter
    # print(f"{t}Starting... {name}")
    ss = tmr()
    counter += 1
    a = f()
    counter -= 1
    print(f"{t}{name} took {tmr() - ss} seconds")
    return a
