def d1(data, part=1):
    data = data.strip('\n')
    res = 0
    indx = 0
    while indx < len(data):
        if part == 1:
            if data[indx] == data[indx - 1]:
                res += int(data[indx])
        else:
            if data[indx] == data[indx - (len(data) / 2)]:
                res += int(data[indx])
        indx += 1
    return res


def d2(data, part=1):
    data = [e.split() for e in data.split('\n')]
    checksum = 0
    for row in data:
        if not row:
            continue
        row = list(map(int, row))
        if part == 1:
            checksum += max(row) - min(row)
            continue
        else:
            i1 = 0
            while i1 < len(row):
                i2 = i1 + 1
                while i2 < len(row):
                    if row[i1] % row[i2] == 0:
                        checksum += row[i1] / row[i2]
                    elif row[i2] % row[i1] == 0:
                        checksum += row[i2] / row[i1]
                    i2 += 1
                i1 += 1
    return checksum


def d3(part=1):
    data = 368078
    if part == 1:
        if data == 1:
            return 0
        side = 1
        dist = 0
        while side ** 2 < data:
            internal_area = side ** 2
            side += 2
            dist += 1
        pos = (data - internal_area) % (side - 1)
        dist += abs(pos - side // 2)
        return dist
    else:
        def find_val(x, y):
            val = 0
            for point in grid:
                if abs(point[0] - x) <= 1 and abs(point[1] - y) <= 1:
                    val += grid[point]
            return val
        grid = {(0, 0): 1}
        corners = 1
        x = 1
        y = 0
        done = False
        while not done:
            while x < corners:
                val = find_val(x, y)
                if val > data:
                    done = True
                    break
                grid[(x, y)] = val
                x += 1
            while y < corners:
                val = find_val(x, y)
                if val > data:
                    done = True
                    break
                grid[(x, y)] = val
                y += 1
            while x > -corners:
                val = find_val(x, y)
                if val > data:
                    done = True
                    break
                grid[(x, y)] = val
                x -= 1
            while y > -corners:
                val = find_val(x, y)
                if val > data:
                    done = True
                    break
                grid[(x, y)] = val
                y -= 1
            corners += 1
        return val


def d4(data, part=1):
    from collections import Counter
    num_valid = 0
    phraselist = data.split('\n')[:-1]
    for phrase in phraselist:
        words = phrase.split(' ')
        if part == 1:
            count = Counter(words)
            if all(v == 1 for v in count.itervalues()):
                num_valid += 1
        else:
            valid = True
            i1 = 0
            while i1 < len(words) - 1:
                i2 = i1 + 1
                while i2 < len(words):
                    if Counter(words[i1]) == Counter(words[i2]):
                        valid = False
                    i2 += 1
                i1 += 1
            if valid:
                num_valid += 1
    return num_valid


def d5(data, part=1):
    data = [int(e) for e in data.split('\n')[:-1]]
    steps = 0
    indx = 0
    while -1 < indx < len(data):
        shift = data[indx]
        if part == 1:
            data[indx] += 1
        else:
            if shift > 2:
                data[indx] -= 1
            else:
                data[indx] += 1
        indx += shift
        steps += 1
    return steps


def d6(data, part=1):
    banks = [int(e) for e in data.split()]
    cycles = 0
    visited = set()
    visited_d = {}
    start_counting = False
    done = False
    while not done:
        max_val = max(banks)
        indx = banks.index(max_val)
        banks[indx] = 0
        if indx == len(banks) - 1:
            indx = 0
        else:
            indx += 1
        while max_val > 0:
            banks[indx] += 1
            if indx == len(banks) - 1:
                indx = 0
            else:
                indx += 1
            max_val -= 1
        if part == 1:
            if tuple(banks) in visited:
                done = True
        else:
            if tuple(banks) in visited_d and visited_d[tuple(banks)] > 1:
                done = True
                return cycles
            if tuple(banks) in visited_d:
                start_counting = True
        visited.add(tuple(banks))
        if tuple(banks) in visited_d:
            visited_d[tuple(banks)] += 1
        else:
            visited_d[tuple(banks)] = 1
        if part == 1:
            cycles += 1
        else:
            if start_counting:
                cycles += 1
    return cycles


def d7(data):
    def get_weight(node):
        result = weights[node]
        child_weights = []
        for child in tower[node]:
            w = get_weight(child)
            child_weights.append(w)
            result += w
        if len(child_weights) > 0:
            median = sorted(child_weights)[len(child_weights) // 2]
            if max(child_weights) > median:
                big_node = tower[node][child_weights.index(max(child_weights))]
                diff = max(child_weights) - median
                print(big_node, weights[big_node], weights[big_node] - diff)
        return result
    bottom = None
    tree = {}
    tower = {}
    nodes = []
    weights = {}
    # nodes = [node]
    # weights = {node: weight}
    # tree = {child: parent, ...}
    # tower = {parent: [children]}
    for row in data.split('\n')[:-1]:
        node = row.split(' ')[0]
        weight = int(row.split(' ')[1].strip('(').strip(')'))
        nodes.append(node)
        weights[node] = weight
        tower[node] = []
        if '->' in row:
            parent = row.split('->')[0].split(' ')[0]
            children = [c.strip() for c in row.split('->')[1].split(', ')]
            tower[parent] = children
            for c in children:
                tree[c] = parent
    for node in nodes:
        if node not in tree:
            bottom = node
    get_weight(bottom)
    return bottom


def d8(data):
    registers = {}
    t_max = 0
    for row in data.split('\n')[:-1]:
        words = row.split(' ')
        target = words[0]
        direction = words[1]
        num = int(words[2])
        check = words[4]
        check_comm = ''.join(words[5:])
        if target not in registers:
            registers[target] = 0
        if check not in registers:
            registers[check] = 0
        if eval('{0}{1}'.format(registers[check], check_comm)):
            if direction[0] == 'i':
                registers[target] += num
            if direction[0] == 'd':
                registers[target] -= num
        if max(registers.values()) > t_max:
            t_max = max(registers.values())
    largest = max(registers.values())
    return largest, t_max


def d9(data):
    score = 0
    group = 0
    indx = 0
    garbage = False
    trash = 0
    while indx < len(data[:-1]):
        if data[indx] == '!':
            indx += 1
        else:
            if not garbage and data[indx] == '{':
                score += 1 + group
                group += 1
            if not garbage and data[indx] == '}':
                group -= 1
            if garbage and data[indx] != '>':
                trash += 1
            if not garbage and data[indx] == '<':
                garbage = True
            if garbage and data[indx] == '>':
                garbage = False
        indx += 1
    return score, trash


def d10(data, part=1):
    if part == 1:
        data = [int(c) for c in data[:-1].split(',')]
        ctr = 63
    else:
        data = [ord(c) for c in data[:-1]]
        data += [17, 31, 73, 47, 23]
        ctr = 0
    li = [i for i in range(0, 256)]
    pos = 0
    skip = 0
    while ctr < 64:
        for length in data:
            if length > len(li):
                continue
            if pos + length > len(li):
                end = (pos + length) % len(li)
                sub = li[pos:] + li[:end]
                sub = sub[::-1]
                li[pos:] = sub[:len(li[pos:])]
                li[:end] = sub[len(li[pos:]):]
            else:
                li[pos:pos + length] = li[pos:pos + length][::-1]
            pos = (pos + length + skip) % len(li)
            skip += 1
        ctr += 1
    if part == 1:
        result = li[0] * li[1]
        return result
    else:
        import numpy
        dense = [0] * 16
        start = 0
        for interval in range(16, 257, 16):
            arr = numpy.array(li[start:interval])
            dense_val = numpy.bitwise_xor.reduce(arr)
            dense_indx = start / 16
            dense[dense_indx] = dense_val
            start = interval
        result = ''
        for num in dense:
            result += hex(int(num))[2:].zfill(2)
        return result


def d11(data, part=2):
    # http://keekerdc.com/2011/03/hexagon-grids-coordinate-systems-and-distance-calculations/
    x, y = 0, 0
    max_dist = 0
    for move in data[:-1].split(','):
        if move == 'n':
            x += 1
        if move == 's':
            x -= 1
        if move == 'nw':
            y += 1
        if move == 'se':
            y -= 1
        if move == 'ne':
            x += 1
            y -= 1
        if move == 'sw':
            x -= 1
            y += 1
        z = - x - y
        dist = max(x - 0, y - 0, z - 0)
        if dist > max_dist:
            max_dist = dist
    return dist, max_dist


def d12(data):
    pipes = {}
    for row in data[:-1].split('\n'):
        source = row.split(' <-> ')[0]
        dest = [i.strip() for i in row.split(' <-> ')[1].split(',')]
        pipes[source] = dest
    group = set()
    group.add('0')

    def add_children(li):
        for elem in li:
            if elem not in group:
                group.add(elem)
                add_children(pipes[elem])
        return None

    add_children(pipes['0'])
    count = len(group)
    num_groups = 1
    for elem in pipes:
        if elem in group:
            continue
        add_children(pipes[elem])
        num_groups += 1
    return count, num_groups


def d13(data):
    scanners = {}
    for scanner in data[:-1].split('\n'):
        scanners[int(scanner.split(':')[0])] = int(scanner.split(':')[1])
    severity = 0
    pico = 0
    while pico <= max(scanners.keys()):
        if pico in scanners:
            if pico % ((scanners[pico] - 1) * 2) == 0:
                severity += pico * scanners[pico]
        pico += 1

    delay = 1
    done = False
    while not done:
        severity = 0
        pico = 0
        caught = False
        while pico <= max(scanners.keys()):
            if pico in scanners:
                if (pico + delay) % ((scanners[pico] - 1) * 2) == 0:
                    caught = True
                    severity += pico * scanners[pico]
            pico += 1
        if not caught:
            done = True
        else:
            delay += 1
    return severity, delay


def d14():
    data = 'hwlqcszp'
    grid = []
    count = 0
    for row in range(0, 128):
        knothash = d10('{0}-{1}\n'.format(data, row), part=2)
        binaries = [bin(int(c, 16))[2:].zfill(4) for c in knothash]
        grid.append(''.join(binaries))
        count += ''.join(binaries).count('1')
    count = 0
    unseen = []
    for x, row in enumerate(grid):
        for y, char in enumerate(row):
            if char == '1':
                unseen.append((x, y))
    while unseen:
        queued = [unseen[0]]
        while queued:
            (x, y) = queued.pop()
            if (x, y) in unseen:
                unseen.remove((x, y))
                queued += [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]
        count += 1
    return count


def d15(part=1):
    def genA(num):
        res = (num * 16807) % 2147483647
        if part == 1 or res % 4 == 0:
            return res
        else:
            return genA(res)

    def genB(num):
        res = (num * 48271) % 2147483647
        if part == 1 or res % 8 == 0:
            return res
        else:
            return genB(res)

    count = 0
    valA = 116
    valB = 299
    if part == 1:
        iterations = 40000000
    else:
        iterations = 5000000
    for i in range(0, iterations):
        valA = genA(valA)
        valB = genB(valB)
        if (valA & 0xFFFF) == (valB & 0xFFFF):
            count += 1
    return count


def d16(data):
    # seen = set()
    programs = [chr(n + 96) for n in range(1, 17)]
    moves = data[:-1].split(',')
    for i in range(0, 1000000000 % 60):
        for move in moves:
            if move[0] == 's':
                spin = int(move[1:])
                programs = programs[-spin:] + programs[:-spin]
            else:
                A = move[1:].split('/')[0]
                B = move[1:].split('/')[1]
                if move[0] == 'x':
                    indx_A = int(A)
                    indx_B = int(B)
                else:
                    indx_A = programs.index(A)
                    indx_B = programs.index(B)
                temp = programs[indx_A]
                programs[indx_A] = programs[indx_B]
                programs[indx_B] = temp
        """if tuple(programs) in seen:
            print(i)
        seen.add(tuple(programs))"""
        # Seen at 60
    return ''.join(programs)


def d17(part=1):
    data = 343
    buff = [0]
    indx = 0
    if part == 1:
        for i in range(1, 2018):
            indx = (indx + data) % len(buff)
            buff.insert(indx + 1, i)
            indx += 1
        return buff[buff.index(2017) + 1]
    else:
        for i in range(1, 50000001):
            indx = (indx + data) % i + 1
            if indx == 1:
                res = i
        return res


def d18(data, part=1):
    commands = []
    last_freq = 0
    for row in data[:-1].split('\n'):
        if len(row.split()) > 2:
            comm, target, val = row.split()
        else:
            comm, target = row.split()
            val = ''
        commands.append((comm, target, val))
    if part == 1:
        registers = {chr(n + 96): 0 for n in range(1, 27)}
        indx = 0
        while 0 <= indx < len(commands):
            command = commands[indx]
            comm = command[0]
            target = command[1]
            try:
                val = int(command[2])
            except ValueError:
                val = command[2]
            if comm == 'snd':
                last_freq = registers[target]
            if comm == 'set':
                if val in registers:
                    registers[target] = registers[val]
                else:
                    registers[target] = val
            if comm == 'add':
                if val in registers:
                    registers[target] += registers[val]
                else:
                    registers[target] += val
            if comm == 'mul':
                if val in registers:
                    registers[target] *= registers[val]
                else:
                    registers[target] *= val
            if comm == 'mod':
                if val in registers:
                    registers[target] %= registers[val]
                else:
                    registers[target] %= val
            if comm == 'rcv' and val != 0:
                return last_freq
            if comm == 'jgz' and target in registers and registers[target] > 0:
                indx += val
            else:
                indx += 1
    else:
        # Not my code
        from collections import defaultdict

        f = open("d18.txt",'r')
        instr = [line.split() for line in f.read().strip().split("\n")]
        f.close()

        d1 = defaultdict(int) # registers for the programs
        d2 = defaultdict(int)
        d2['p'] = 1
        ds = [d1, d2]

        def get(s):
            if s in "qwertyuiopasdfghjklzxcvbnm":
                return d[s]
            return int(s)

        tot = 0

        ind = [0, 0]         # instruction indices for both programs
        snd = [[], []]       # queues of sent data (snd[0] = data that program 0 has sent)
        state = ["ok", "ok"] # "ok", "r" for receiving, or "done"
        pr = 0     # current program
        d = ds[pr] # current program's registers
        i = ind[0] # current program's instruction index
        while True:
            if instr[i][0] == "snd": # send
                if pr == 1: # count how many times program 1 sends
                    tot += 1
                snd[pr].append(get(instr[i][1]))
            elif instr[i][0] == "set":
                d[instr[i][1]] = get(instr[i][2])
            elif instr[i][0] == "add":
                d[instr[i][1]] += get(instr[i][2])
            elif instr[i][0] == "mul":
                d[instr[i][1]] *= get(instr[i][2])
            elif instr[i][0] == "mod":
                d[instr[i][1]] %= get(instr[i][2])
            elif instr[i][0] == "rcv":
                if snd[1 - pr]: # other program has sent data
                    state[pr] = "ok"
                    d[instr[i][1]] = snd[1 - pr].pop(0) # get data
                else: # wait: switch to other prog
                    if state[1 - pr] == "done":
                        break # will never recv: deadlock
                    if len(snd[pr]) == 0 and state[1 - pr] == "r":
                        break # this one hasn't sent anything, other is recving: deadlock
                    ind[pr] = i   # save instruction index
                    state[pr] = "r" # save state
                    pr = 1 - pr   # change program
                    i = ind[pr] - 1 # (will be incremented back)
                    d = ds[pr]    # change registers
            elif instr[i][0] == "jgz":
                if get(instr[i][1]) > 0:
                    i += get(instr[i][2]) - 1
            i += 1
            if not 0 <= i < len(instr):
                if state[1 - pr] == "done":
                    break # both done
                state[pr] = "done"
                ind[pr] = i  # swap back since other program's not done
                pr = 1 - pr
                i = ind[pr]
                d = ds[pr]

        print tot
        # My code (doesn't work...)
        """reg0 = {chr(n + 96): 0 for n in range(1, 27)}
        reg1 = {chr(n + 96): 0 for n in range(1, 27)}
        reg1['p'] = 1
        indx0, indx1 = 0, 0
        queue0, queue1 = [], []
        done0, done1 = False, False
        count = 0
        while not done0 and not done1:
            if 0 <= indx0 < len(commands):
                comm0, targ0, val0 = commands[indx0]
                try:
                    val0 = int(val0)
                except ValueError:
                    pass
                if comm0 == 'snd':
                    queue1.append(reg0[targ0])
                if comm0 == 'set':
                    if val0 in reg0:
                        reg0[targ0] = reg0[val0]
                    else:
                        reg0[targ0] = val0
                if comm0 == 'add':
                    if val0 in reg0:
                        reg0[targ0] += reg0[val0]
                    else:
                        reg0[targ0] += val0
                if comm0 == 'mul':
                    if val0 in reg0:
                        reg0[targ0] *= reg0[val0]
                    else:
                        reg0[targ0] *= val0
                if comm0 == 'mod':
                    if val0 in reg0:
                        reg0[targ0] %= reg0[val0]
                    else:
                        reg0[targ0] %= val0
                if comm0 == 'rcv' and val0 != 0:
                    if len(queue0) == 0:
                        done0 = True
                    else:
                        if val0 in reg0:
                            reg0[targ0] = queue0.pop(0)
                        else:
                            reg0[targ0] = queue0.pop(0)
                if comm0 == 'jgz' and targ0 in reg0 and reg0[targ0] > 0:
                    if val0 in reg0:
                        indx0 += reg0[val0]
                    else:
                        indx0 += val0
                else:
                    indx0 += 1
            if 0 <= indx1 < len(commands):
                comm1, targ1, val1 = commands[indx1]
                try:
                    val1 = int(val1)
                except ValueError:
                    pass
                if comm1 == 'snd':
                    count += 1
                    queue0.append(reg1[targ1])
                if comm1 == 'set':
                    if val1 in reg1:
                        reg1[targ1] = reg1[val1]
                    else:
                        reg1[targ1] = val1
                if comm1 == 'add':
                    if val1 in reg1:
                        reg1[targ1] += reg1[val1]
                    else:
                        reg1[targ1] += val1
                if comm1 == 'mul':
                    if val1 in reg1:
                        reg1[targ1] *= reg1[val1]
                    else:
                        reg1[targ1] *= val1
                if comm1 == 'mod':
                    if val1 in reg1:
                        reg1[targ1] %= reg1[val1]
                    else:
                        reg1[targ1] %= val1
                if comm1 == 'rcv' and val1 != 0:
                    if len(queue1) == 0:
                        done1 = True
                    else:
                        if val1 in reg1:
                            reg1[targ1] = queue1.pop(0)
                        else:
                            reg1[targ1] = queue1.pop(0)
                if comm1 == 'jgz' and targ1 in reg1 and reg1[targ1] > 0:
                    if val1 in reg1:
                        indx1 += reg1[val1]
                    else:
                        indx1 += val1
                else:
                    indx1 += 1
        return count, indx0, indx1"""


def d19(data):
    steps = 0
    grid = []
    for row in data:
        grid.append([c for c in row[:-1]])
    res = []
    coord = [grid[0].index('|'), 0]
    move = (0, 1)
    done = False
    while not done:
        next_coord = [sum(x) for x in zip(coord, move)]
        if grid[next_coord[1]][next_coord[0]].isalpha():
            res.append(grid[next_coord[1]][next_coord[0]])
            coord = [sum(x) for x in zip(coord, move)]
        if grid[next_coord[1]][next_coord[0]] == '+':
            if move == (0, 1) or move == (0, -1):
                if grid[next_coord[1]][next_coord[0] - 1] == ' ':
                    move = (1, 0)
                else:
                    move = (-1, 0)
            else:
                if grid[next_coord[1] - 1][next_coord[0]] == ' ':
                    move = (0, 1)
                else:
                    move = (0, -1)
        coord = next_coord
        steps += 1
        if grid[next_coord[1]][next_coord[0]] == ' ':
            done = True
    return ''.join(res), steps


def d20(data):
    # 345 too high
    # 197 too high
    particles = {}
    ctr = 0
    for row in data[:-1].split('\n'):
        p, v, a = row.split(' ')
        pxyz = [int(e) for e in p[p.find('<') + 1:p.find('>')].split(',')]
        vxyz = [int(e) for e in v[v.find('<') + 1:v.find('>')].split(',')]
        axyz = [int(e) for e in a[a.find('<') + 1:a.find('>')].split(',')]
        particles[ctr] = {'p': pxyz, 'v': vxyz, 'a': axyz}
        ctr += 1

    def update_position(num):
        particles[num]['v'] = [sum(e) for e in zip(particles[num]['v'],
                                                   particles[num]['a'])]
        particles[num]['p'] = [sum(e) for e in zip(particles[num]['p'],
                                                   particles[num]['v'])]

    def get_dist(num):
        return sum([abs(e) for e in particles[num]['p']])

    destroyed = []
    for _ in range(0, 500):
        dists = []
        for particle in particles:
            update_position(particle)
            dists.append(get_dist(particle))

        positions = {}
        for particle in particles:
            if particle in destroyed:
                continue
            position = tuple(particles[particle]['p'])
            if position not in positions:
                positions[position] = [particle]
            else:
                positions[position].append(particle)
        for pos in positions:
            if len(positions[pos]) > 1:
                destroyed += positions[pos]
    return dists.index(min(dists)), len(particles) - len(destroyed)


def d21(data):
    import numpy as np
    rules = {}
    for row in data.split('\n')[:-1]:
        src, trg = [e.split('/') for e in row.split(' => ')]
        src = np.array([list(r) for r in src])
        trg = np.array([list(r) for r in trg])
        # Original matrix
        rules[src.tobytes()] = trg
        # Rotated matrices
        rules[np.rot90(src, k=1).tobytes()] = trg
        rules[np.rot90(src, k=2).tobytes()] = trg
        rules[np.rot90(src, k=3).tobytes()] = trg
        # Flipped (and rotated) matrices
        rules[np.flip(src, axis=1).tobytes()] = trg
        # Rotated matrices
        rules[np.rot90(np.flip(src, axis=1), k=1).tobytes()] = trg
        rules[np.rot90(np.flip(src, axis=1), k=2).tobytes()] = trg
        rules[np.rot90(np.flip(src, axis=1), k=3).tobytes()] = trg
    
    # Starting grid
    grid = np.array([['.','#','.'], ['.','.','#'], ['#','#','#']])
    for _ in range(0, 18):
        if len(grid) % 2 == 0:
            tgrid = False
            for row in range(0, len(grid), 2):
                rgrid = np.array([[]])
                for col in range(0, len(grid), 2):
                    subset = grid[row:row + 2, col:col + 2]
                    if col == 0:
                        rgrid = rules[subset.tobytes()]
                    else:
                        rgrid = np.concatenate((rgrid, rules[subset.tobytes()]), axis=1)
                if row == 0:
                    tgrid = rgrid
                else:
                    tgrid = np.concatenate((tgrid, rgrid), axis=0)
        else:
            tgrid = False
            for row in range(0, len(grid), 3):
                rgrid = np.array([[]])
                for col in range(0, len(grid), 3):
                    subset = grid[row:row + 3, col:col + 3]
                    if col == 0:
                        rgrid = rules[subset.tobytes()]
                    else:
                        rgrid = np.concatenate((rgrid, rules[subset.tobytes()]), axis=1)
                if row == 0:
                    tgrid = rgrid
                else:
                    tgrid = np.concatenate((tgrid, rgrid), axis=0)
        grid = tgrid
    return (grid == '#').sum()


if __name__ == '__main__':
    with open('d21.txt', 'r') as fid:
        data = ''.join(fid.readlines())
    print(d21(data))
