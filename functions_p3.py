import numpy as np
import subprocess
import sys
import math
import itertools
import os
import tempfile
import re
import json
import copy
import GLOBAL

from vectors import RealVector, DomainElementFactory, interpolate

from datetime import date
from sklearn import svm

temp_dir = None


class Chronicle:
    """Class for a chronicle pattern modeling
    """
    nchro = 0

    def __init__(self, parent_dataset=None):
        self.multiset = list()  # sorted list of items (e)
        self.tconst = {}  # temporal constraints
        self.cid = Chronicle.nchro  # chronicle id
        Chronicle.nchro += 1
        self.svmkernel = False
        self.classifier = None
        self.dataset = parent_dataset
        self.wracc = -1
        self.tp = -1
        self.fp = -1
        self.label = None
        self.support = {}

    def set_parent_dataset(self, parent_dataset):
        self.dataset = parent_dataset

    def add_item(self, item):
        """Add an item to the chronicle and return the id of the added event
        The function creates all infinite constraints, without variability
        - the id of the event correspond to the order of added items
        """
        self.multiset.append(item)
        id = len(self.multiset) - 1
        for i in range(len(self.multiset) - 1):
            self.tconst[(i, id)] = (
                DomainElementFactory.negative_infinity(),
                DomainElementFactory.positive_infinity()
            )
        return id

    def add_constraint(self, ei, ej, constr):
        """Add a constraint-template to the chronicle pattern
        - ei, ej: index of the events in the multiset
        - constr: a 4-tuple (mu-start, var-start, mu-end, var-end) of the mean and variance of temporal constraint
        """
        if not type(constr) is tuple:
            print("[ERROR] constraint must be a tuple (=> constraint not added)", file=sys.stderr)
            return

        if len(constr) != 2:
            print("[ERROR] constraint must have 4 values (=> constraint not added)", file=sys.stderr)
            return

        try:
            self.tconst[(ei, ej)] = constr
        except IndexError:
            print("[ERROR] index_error (=> constraint not added)", file=sys.stderr)

    def is_occur(self, occ):
        n = len(self.multiset)
        sumn = n * (n - 1) / 2
        for key, tc in self.tconst.items():
            ind = sumn - (n - key[0]) * (n - key[0] - 1) / 2 + key[1] - key[0] - 1
            if not occ[int(ind)].is_inside_bounds(tc[0], tc[1]):
                return False
        return True

    def occur(self, bag):
        for occ in bag:
            if self.is_occur(occ):
                return True
        return False

    def occur_in(self, base):
        m = self.get_map_multiset()
        keys = sorted(m.keys())
        if keys[-1] >= len(base):
            return False
        occs = [[occ for occ in itertools.combinations(base[k][0], m[k])] for k in keys]
        if all(len(x) > 0 for x in occs):
            for o in itertools.product(*occs):
                combinations = itertools.combinations([a for b in o for a in b], 2)
                occur = [q - p for p, q in combinations]
                if self.is_occur(occur):
                    return True
        return False

    def set_support(self, base):
        for label in base.db:
            if label not in self.support:
                self.support[label] = []
            for seq in range(len(base.db[label][0])):
                if self.occur_in([[item[seq]] for item in base.db[label]]):
                    self.support[label].append(seq)

    def set_wracc(self, base, seq_use=None):
        n = base.get_nb_seq()
        tp = self.tp
        if seq_use:
            tp = 0
            for seq in self.support[self.label]:
                if seq in seq_use[self.label]:
                    tp += 1/float(1+seq_use[self.label][seq])
                else:
                    tp += 1

        pcond = float(tp + self.fp) / float(n)
        pclasscond = float(tp) / float(tp + self.fp)
        pclass = float(base.get_nb_seq(self.label)) / float(n)
        self.wracc = pcond * (pclasscond - pclass)
        return self.wracc

    def get_long_multiset(self, size):
        res = [0 for _ in range(size)]
        for i in self.multiset:
            res[i] += 1
        return res

    def get_map_multiset(self):
        map = {}
        for i in self.multiset:
            if i in map:
                map[i] += 1
            else:
                map[i] = 1
        return map

    def get_growth(self, pos, neg):
        fp = 0
        fn = 0
        for bag in pos:
            if self.occur(bag):
                fp += 1
        for bag in neg:
            if self.occur(bag):
                fn += 1
        return fp, fn

    def get_constraint(self, i):
        x = 0
        y = i
        while y - (len(self.multiset) - x - 1) >= 0:
            y -= len(self.multiset) - x - 1
            x += 1
        return x, (y + x + 1)

    @staticmethod
    def parse_cpp(f, reverse=None):
        """
        Parse a list of chronicles with same format as Extract cpp output
        :param f: The file to parse
        :param reverse: A dictionary to reverse item names (None for no reverse)
        :return: The parsed chronicle list
        """
        reg_start = "^C(?:[0-9]*): \{(\[.+\])\}$"
        reg_tcs_new = "^([0-9]+),(?: ?)([0-9]+): \((<(?:-?(?:inf|[0-9.]+),?)+>), (<(?:-?(?:inf|[0-9.]+),?)+>)\)$"
        reg_freq = "^f: ([0-9.]*)/([0-9.]*)$"

        res = []
        tmp = None
        with open(f, 'r') as fin:
            i = 0
            order = {}
            for line in fin:
                l = line.replace('\r\n', '').replace('\n', '')
                m_set = re.match(reg_start, l)
                if m_set is not None:
                    order = {}
                    ml = m_set.group(1)
                    ml = ml.replace("'", '"')
                    try:
                        ml = json.loads(ml)
                    except:
                        print("[ERROR] Fichier: " + f + "\n" + l + "\n" + "ligne: " + str(i), file=sys.stderr)
                        sys.exit()
                    if tmp is not None:
                        res.append(tmp)
                    tmp = [Chronicle(), '', 0, 0]
                    for x in ml:
                        if reverse:
                            tmp[0].add_item(reverse[str(x)])
                        else:
                            tmp[0].add_item(str(x))
                    if reverse:
                        arr_order = sorted(range(len(tmp[0].multiset)), key = lambda k: tmp[0].multiset[k])
                        for i in range(len(arr_order)):
                            order[arr_order[i]] = i
                        tmp[0].multiset = sorted(tmp[0].multiset)
                else:
                    m_set = re.match(reg_tcs_new, l)
                    if m_set is not None:
                        source = int(m_set.group(1))
                        target = int(m_set.group(2))
                        inf = DomainElementFactory.parse_from_cpp(m_set.group(3))
                        sup = DomainElementFactory.parse_from_cpp(m_set.group(4))
                        if order:
                            source = int(order[source])
                            target = int(order[target])
                            if source > target:
                                source, target = target, source
                                inf, sup = -sup, -inf
                        tmp[0].add_constraint(source, target, (inf, sup))
                    else:
                        m_set = re.match(reg_freq, l)
                        if m_set is not None:
                            tmp[2] = float(m_set.group(1))
                            tmp[3] = float(m_set.group(2))

                i += 1
        if tmp is not None:
            res.append(tmp)
        return res

    @staticmethod
    def parse(f, f_label=False):
        """
        Parse a list of chronicles
        :param f: The file to parse
        :param f_label: use f instead of sup\(c,pos\)/sup\(c,neg\) to recognize freq line
        :return: The parsed chronicle list
        """
        reg_start = "^C(?:[0-9]*): \{(\[.+\])\}$"
        reg_tcs = "^([0-9]+),(?: ?)([0-9]+): \((-?(?:inf|[0-9.]+)), (-?(?:inf|[0-9.]+))\)$"
        reg_label = "^class: (.+)$"
        reg_freq = "^sup\(c,pos\)/sup\(c,neg\): ([0-9]*)/([0-9]*)$"
        if f_label:
            reg_freq = "^f: ([0-9]*)/([0-9]*)$"
        res = []
        tmp = None
        with open(f, 'r') as fin:
            i = 0
            for line in fin:
                l = line.replace('\r\n', '').replace('\n', '')
                m_set = re.match(reg_start, l)
                if m_set is not None:
                    ml = m_set.group(1)
                    ml = ml.replace("'", '"')
                    try:
                        ml = json.loads(ml)
                    except:
                        print("[ERROR] Fichier: " + f + "\n" + l + "\n" + "ligne: " + str(i), file=sys.stderr)
                        sys.exit()
                    if tmp is not None:
                        res.append(tmp)
                    tmp = [Chronicle(), '', 0, 0]
                    for x in ml:
                        tmp[0].add_item(str(x))
                else:
                    m_set = re.match(reg_tcs, l)
                    if m_set is not None:
                        source = m_set.group(1)
                        target = m_set.group(2)
                        inf = m_set.group(3)
                        if inf == "-inf":
                            inf = -float("inf")
                        sup = m_set.group(4)
                        if sup == "inf":
                            sup = float("inf")
                        tmp[0].add_constraint(float(source), float(target), (float(inf), float(sup)))
                    else:
                        m_set = re.match(reg_label, l)
                        if m_set is not None:
                            tmp[1] = m_set.group(1)
                        else:
                            m_set = re.match(reg_freq, l)
                            if m_set is not None:
                                tmp[2] = float(m_set.group(1))
                                tmp[3] = float(m_set.group(2))

                i += 1
        if tmp is not None:
            res.append(tmp)
        return res

    def __getitem__(self, i):
        """return the item at position i in the multiset if i is an integer
        and return the constraint between i[0] and i[1] if i is a couple
        """
        if not type(i) is tuple:
            return self.multiset[i]
        else:
            return self.tconst[(min(i[0], i[1]), max(i[0], i[1]))]

    def __len__(self):
        return len(self.multiset)

    def __str__(self):
        s = ""
        if not self.dataset:
            # print "Warning, I'm printing item ids instead of item names because there is no parent dataset defined. Use set_parent_dataset"
            s = "C" + str(self.cid) + ": {" + str(self.multiset) + "}\n"
            for i in range(len(self.multiset)):
                for j in range(i + 1, len(self.multiset)):
                    s += str(i) + "," + str(j) + ": " + self.print_tconst(self.tconst[(i, j)]) + "\n"
        else:
            item_mapping = self.dataset.item_mapping
            multiset = self.multiset

            s = "C" + str(self.cid) + ": {"
            for i in range(len(self.multiset)):
                s += str(item_mapping[multiset[i]]) + "(" + str(i) + "), "
            s += "}\n"
            for i in range(len(self.multiset)):
                for j in range(i + 1, len(multiset)):
                    s += item_mapping[multiset[i]] + "(" + str(i) + ")," + item_mapping[multiset[j]] + "(" + str(
                        j) + "): " + self.print_tconst(self.tconst[(i, j)]) + "\n"
        return s

    def print_tconst(self, tconst):
        out = "("

        for i in range(0, len(tconst)):
            out += str(tconst[i])

            if i < len(tconst) - 1:
                out += ", "

        out += ")"

        return out


class McDataset:
    def __init__(self, item_mapping=[]):
        self.item_mapping = item_mapping
        self.reverse_item_mapping = {}
        self.db = {}
        self.db_seq = {}
        self.max_occ = {}
        self.group = {}
        self.inv_group = {}

    def __deepcopy__(self):
        new = type(self)()
        new.item_mapping = copy.deepcopy(self.item_mapping)
        new.reverse_item_mapping = copy.deepcopy(self.reverse_item_mapping)
        new.db = copy.deepcopy(self.db)
        new.db_seq = copy.deepcopy(self.db_seq)
        new.max_occ = copy.deepcopy(self.max_occ)
        new.group = copy.deepcopy(self.group)
        new.inv_group = copy.deepcopy(self.inv_group)
        return new

    def del_doubles(self):
        for label in self.db:
            for i in range(len(self.db[label])):
                for seq in range(len(self.db[label][i])):
                    self.db[label][i][seq] = sorted(list(set(self.db[label][i][seq])))

    def get_nb_seq(self, label=None):
        nb = 0
        if not label:
            for l in self.db:
                nb += len(self.db[l][0])
        elif label in self.db:
            nb += len(self.db[label][0])
        return nb

    def create_db_seq(self, only_group=False, store=False):
        db_seq = {}
        for label in self.db:
            db_seq[label] = []
            for i in range(self.get_nb_seq(label)):
                tmp = []
                for j in range(len(self.db[label])):
                    for t in self.db[label][j][i]:
                        tmp.append((t, j))
                tmp = sorted(tmp)
                db_seq[label].append([])
                for e in tmp:
                    name = e[1]
                    group = False
                    if int(name) < len(self.item_mapping):
                        name = self.item_mapping[name]
                    else:
                        group = True
                        gname = name
                        if len(self.item_mapping) % 2 == 0:
                            par = name % 2
                            gname = name - par
                        else:
                            par = (name+1) % 2
                            gname = name - par
                        name = "group_" + str(self.item_mapping[self.group[gname]])
                        if par > 0:
                            name += "_fin"
                        else:
                            name += "_deb"

                    if not only_group or group:
                        db_seq[label][-1].append(name)
                        db_seq[label][-1].append(e[0])
        if store:
            self.db_seq = db_seq
        return db_seq

    def regroup_events(self, minn, maxdelay):
        for label in self.db:
            for _ in range(len(self.db[label]), len(self.item_mapping) + len(self.inv_group)*2):
                self.db[label].append([])
                for __ in range(len(self.db[label][0])):
                    self.db[label][-1].append([])

            init_len = len(self.item_mapping)
            for i in range(init_len):
                for seq in range(len(self.db[label][i])):
                    tmp = []
                    suppress = []
                    if self.item_mapping[i] == "N02BE01" and len(self.db[label][i][seq]) > 0:
                        suppress.append((self.db[label][i][seq][0], self.db[label][i][seq][-1]))
                    else:
                        for t in range(len(self.db[label][i][seq])):
                            if not tmp:
                                tmp.append(t)
                            else:
                                if self.db[label][i][seq][t] - self.db[label][i][seq][tmp[-1]] <= maxdelay:
                                    tmp.append(t)
                                else:
                                    if len(tmp) >= minn:
                                        if i not in self.inv_group:
                                            gi = len(self.item_mapping) + len(self.inv_group)*2
                                            self.inv_group[i] = (gi, gi+1)
                                            self.group[gi] = i

                                            for _ in range(len(self.db[label]), gi + 2):
                                                self.db[label].append([])

                                            for _ in self.db[label][i]:
                                                self.db[label][gi].append([])
                                                self.db[label][gi+1].append([])
                                        # On fait l'hypothese qu'il n'existe pas une periode d'exposition du meme type
                                        # deja existante avec un timestamp superieur (attention !)
                                        self.db[label][self.inv_group[i][0]][seq].append(self.db[label][i][seq][tmp[0]])
                                        self.db[label][self.inv_group[i][1]][seq].append(self.db[label][i][seq][tmp[-1]])
                                        suppress.append((tmp[0], tmp[1]))
                                    tmp = []
                        tmp = []
                    step = 0
                    for t in range(len(self.db[label][i][seq])):
                        while step < len(suppress) and t > suppress[step][1]:
                            step += 1
                        if step >= len(suppress) or t < suppress[step][0] or t > suppress[step][1]:
                            tmp.append(self.db[label][i][seq][t])
                    self.db[label][i][seq] = tmp

    def set_reverse_item_mapping(self, reverse_item_mapping):
        self.reverse_item_mapping = reverse_item_mapping

    def shuffle(self):
        for label in self.db:
            permut = np.random.permutation(len(self.db[label][0]))
            self.db[label] = [np.array(item)[permut.tolist()].tolist() for item in self.db[label]]
            if len(self.db_seq) > 0:
                self.db_seq[label] = np.array(self.db_seq[label])[permut.tolist()].tolist()

    def shrink(self, n, label=None):
        if label in self.db:
            for i in range(len(self.db[label])):
                self.db[label][i] = self.db[label][i][:n]
        else:
            for lab in self.db:
                for i in range(len(self.db[lab])):
                    self.db[lab][i] = self.db[lab][i][:n]

    def load_db_line(self, fi, label, use_date=False, store_seq=False):
        if label not in self.db:
            self.db[label] = []
        if store_seq and label not in self.db_seq:
            self.db_seq[label] = []
        with open(fi, 'r') as fin:
            nbseq = 0
            max_occ = 0
            db = self.db[label]
            # ~ event type + <vecsize>-times temporal vector elemeents
            event_size = DomainElementFactory.VECTOR_SIZE + 1
            for line in fin:
                lline = line.split()
                if len(lline) % event_size == 0:
                    if len(lline) > 0:
                        if store_seq:
                            self.db_seq[label].append([])
                        for i in range(0, len(lline), event_size):
                            item = lline[i]
                            time_tuple = tuple()

                            for j in range(1, DomainElementFactory.VECTOR_SIZE + 1):
                                time_tuple = time_tuple + (float(lline[i + j]), )

                            time = RealVector(time_tuple)

                            if use_date:
                                j = time % 100
                                m = int(time / 100) % 100
                                a = int(time / 10000)
                                dbase = date(2009, 1, 1)
                                d = date(a, m, j)
                                time = (d - dbase).days

                            if store_seq:
                                self.db_seq[label][-1].append(item)
                                self.db_seq[label][-1].append(time)
                            if not item in self.reverse_item_mapping:
                                self.reverse_item_mapping[item] = len(self.item_mapping)
                                self.item_mapping.append(item)
                            it = self.reverse_item_mapping[item]
                            for _ in range(len(db), len(self.item_mapping)):
                                db.append([])
                            for _ in range(len(db[it]), nbseq + 1):
                                db[it].append([])
                            db[it][nbseq].append(time)
                            if len(db[it][nbseq]) > max_occ:
                                max_occ = len(db[it][nbseq])
                        nbseq += 1
                    else:
                        raise Exception(interpolate("Input line #{nbseq} in {fi} dataset is not correct length."))
            for item in db:
                for _ in range(len(item), nbseq):
                    item.append([])
            self.max_occ[label] = max_occ

    def write_ms(self, label):
        out = ""
        db = self.db[label]
        max_occ = self.max_occ[label]
        occ_size = int(math.log10(max_occ) + 1)
        if len(db) > 0:
            for seq in range(len(db[0])):
                tmp = ""
                for i in range(len(db)):
                    for x in range(len(db[i][seq])):
                        tmp += str(i + 1) + str(x + 1).zfill(occ_size) + " "
                if tmp is not "":
                    out += tmp[:-1] + "\n"
        return out, occ_size

    def write_seq(self, label, sep_event=" ", sep_inter=" "):
        out = ""
        db = self.db_seq[label]
        for seq in db:
            out += sep_event.join([str(e) + sep_inter + t.format_for_dcm_input() for (e, t) in zip(*[iter(seq)] * 2)]) + "\n"
        return out

    def support(self, m, label):
        db = self.db[label]
        sup = 0
        if sorted(m.keys())[-1] >= len(db):
            return 0
        for seq in range(len(db[0])):
            ok = True
            for k in sorted(m.keys()):
                if len(db[k][seq]) < m[k]:
                    ok = False
                    break
            if ok:
                sup += 1
        return sup

    def discr_ms(self, ms, gmin, l, p):
        others = []
        discr = []
        for m in ms:
            suppos = self.support(m, l)
            supneg = 0
            for label in self.db:
                if label != l:
                    supneg += self.support(m, label)
            if suppos >= gmin * supneg:
                c = Chronicle(self)
                for k in sorted(m.keys()):
                    for _ in range(m[k]):
                        c.add_item(k)
                if p:
                    print(c)

                discr.append((c, l, suppos, supneg))
            else:
                tot = 0
                for k in m.keys():
                    tot += m[k]
                if tot >= 2:
                    others.append(m)
        return others, discr

    def get_bags(self, m, label=None, db=None):
        if not db:
            if not label:
                label = self.db.keys()[0]
            db = self.db[label]
        bags = []
        if sorted(m.keys())[-1] >= len(db):
            return bags
        for i in range(len(db[0])):
            bag = []
            occs = [[occ for occ in itertools.combinations(db[k][i], m[k])] for k in sorted(m.keys())]
            if all(len(x) > 0 for x in occs):
                for o in itertools.product(*occs):
                    bag.append([q - p for p, q in itertools.combinations([a for b in o for a in b], 2)])
            if len(bag) > 0:
                bags.append(bag)
        return bags

    def write_occs(self, m, label):
        out = ""
        bags = self.get_bags(m, label)
        for i, bag in enumerate(bags):
            for occ in bag:
                out += " ".join(str(k) for k in occ)
                out += " pos " + str(i) + "\n"
        bags = []
        for l in self.db:
            if label != l:
                bags += self.get_bags(m, l)
        for i, bag in enumerate(bags):
            for occ in bag:
                out += " ".join(str(k) for k in occ)
                out += " neg " + str(i) + "\n"
        return out

    def get_item_from_id(self, item):
        return self.reverse_item_mapping[item]

    def load_constraints(self, ms, fi):
        ttcs = []
        with open(fi, 'r') as fin:
            re_tcs = "\(([0-9]+), (-inf|-?[0-9]+), (\+inf|-?[0-9]+)\)"
            for line in fin:
                tcs = []
                l = line.replace('\n', '')
                for m_tcs in re.finditer(re_tcs, l):
                    for i in range(0, len(m_tcs.groups()), 4):
                        tcs.append((int(m_tcs.group(i + 1)), float(m_tcs.group(i + 2)), float(m_tcs.group(i + 3))))
                if len(tcs) > 0:
                    ttcs.append(tcs)
        chronicles = []
        for tcs in ttcs:
            c = Chronicle(self)
            for k in sorted(ms.keys()):
                for _ in range(ms[k]):
                    c.add_item(k)
            for tc in tcs:
                inf, sup = c.get_constraint(int(tc[0]))
                c.add_constraint(inf, sup, (float(tc[1]), float(tc[2])))
            chronicles.append(c)
        return chronicles


class Dataset:
    def __init__(self, item_mapping=[]):
        self.item_mapping = item_mapping
        self.reverse_item_mapping = {}
        self.db_pos = []  # positive sequence database
        self.max_occ_pos = 0
        self.db_neg = []  # negative sequence database
        self.max_occ_neg = 0

    def set_reverse_item_mapping(self, reverse_item_mapping):
        self.reverse_item_mapping = reverse_item_mapping

    def shuffle(self):
        permut = np.random.permutation(len(self.db_pos[0]))
        self.db_pos = [np.array(item)[permut.tolist()].tolist() for item in self.db_pos]
        permut = np.random.permutation(len(self.db_neg[0]))
        self.db_neg = [np.array(item)[permut.tolist()].tolist() for item in self.db_neg]

    def select_db(self, pos):
        if pos:
            return self.db_pos
        return self.db_neg

    def load_db_line(self, fi, pos):
        with open(fi, 'r') as fin:
            nbseq = 0
            max_occ = 0
            db = self.select_db(pos)
            for line in fin:
                lline = line.split()
                if len(lline) > 0 and len(lline) % 2 == 0:
                    for i in range(0, len(lline), 2):
                        item = lline[i]
                        time = int(lline[i + 1])
                        if not item in self.reverse_item_mapping:
                            self.reverse_item_mapping[item] = len(self.item_mapping)
                            self.item_mapping.append(item)
                        it = self.reverse_item_mapping[item]
                        for _ in range(len(db), len(self.item_mapping) + 1):
                            db.append([])
                        for _ in range(len(db[it]), nbseq + 1):
                            db[it].append([])
                        db[it][nbseq].append(time)
                        if len(db[it][nbseq]) > max_occ:
                            max_occ = len(db[it][nbseq])
                    nbseq += 1
            for item in db:
                for _ in range(len(item), nbseq):
                    item.append([])
            if pos:
                self.max_occ_pos = max_occ
            else:
                self.max_occ_neg = max_occ

    def write_ms(self, pos):
        out = ""
        db = self.select_db(pos)
        max_occ = self.max_occ_neg
        if pos:
            max_occ = self.max_occ_pos
        occ_size = int(math.log10(max_occ) + 1)
        if len(db) > 0:
            for seq in range(len(db[0])):
                tmp = ""
                for i in range(len(db)):
                    for x in range(len(db[i][seq])):
                        tmp += str(i + 1) + str(x + 1).zfill(occ_size) + " "
                if tmp is not "":
                    out += tmp[:-1] + "\n"
        return out, occ_size

    def support(self, m, pos):
        db = self.select_db(pos)
        sup = 0
        for seq in range(len(db[0])):
            ok = True
            for k in sorted(m.keys()):
                if len(db[k][seq]) < m[k]:
                    ok = False
                    break
            if ok:
                sup += 1
        return sup

    def discr_ms(self, ms, gmin, p):
        others = []
        discr = []
        for m in ms:
            suppos = self.support(m, True)
            supneg = self.support(m, False)
            if suppos >= gmin * supneg:
                c = Chronicle(self)
                for k in sorted(m.keys()):
                    for _ in range(m[k]):
                        c.add_item(k)
                if p:
                    print(str(c) + "sup(c,pos)/sup(c,neg): " + str(suppos) + "/" + str(supneg) + "\n")
                discr.append((c, suppos, supneg))
            else:
                tot = 0
                for k in m.keys():
                    tot += m[k]
                if tot >= 2:
                    others.append(m)
        return others, discr

    def get_bags(self, m, pos=True, db=None):
        if db is None:
            db = self.select_db(pos)
        bags = []
        for i in range(len(db[0])):
            bag = []
            occs = [[occ for occ in itertools.combinations(db[k][i], m[k])] for k in sorted(m.keys())]
            if all(len(x) > 0 for x in occs):
                for o in itertools.product(*occs):
                    bag.append([q - p for p, q in itertools.combinations([a for b in o for a in b], 2)])
            if len(bag) > 0:
                bags.append(bag)
        return bags

    def write_occs(self, m):
        out = ""
        bags = self.get_bags(m, True)
        for i, bag in enumerate(bags):
            for occ in bag:
                out += " ".join(str(k) for k in occ)
                out += " pos " + str(i) + "\n"
        bags = self.get_bags(m, False)
        for i, bag in enumerate(bags):
            for occ in bag:
                out += " ".join(str(k) for k in occ)
                out += " neg " + str(i) + "\n"
        return out

    def get_item_from_id(self, item):
        return self.reverse_item_mapping[item]

    def load_constraints(self, ms, fi):
        ttcs = []
        with open(fi, 'r') as fin:
            re_tcs = "\(([0-9]+), (-inf|-?[0-9]+), (\+inf|-?[0-9]+)\)"
            for line in fin:
                tcs = []
                l = line.replace('\n', '')
                for m_tcs in re.finditer(re_tcs, l):
                    for i in range(0, len(m_tcs.groups()), 4):
                        tcs.append((int(m_tcs.group(i + 1)), float(m_tcs.group(i + 2)), float(m_tcs.group(i + 3))))
                if len(tcs) > 0:
                    ttcs.append(tcs)
        chronicles = []
        for tcs in ttcs:
            c = Chronicle(self)
            for k in sorted(ms.keys()):
                for _ in range(ms[k]):
                    c.add_item(k)
            for tc in tcs:
                inf, sup = c.get_constraint(int(tc[0]))
                c.add_constraint(inf, sup, (float(tc[1]), float(tc[2])))
            chronicles.append(c)
        return chronicles


def mc_generate_arff(name, base, chronicles):
    with open(name, 'w') as fout:
        fout.write('@RELATION features\n\n')
        for i in range(len(chronicles)):
            fout.write('@ATTRIBUTE c' + str(chronicles[i][0].cid) + '\tNUMERIC\n')
        fout.write('@ATTRIBUTE class\t{' + ",".join(base.db.keys()) + '}\n\n@DATA\n')
        for label in base.db:
            for seq in range(len(base.db[label][0])):
                out = ""
                for c in chronicles:
                    if c[0].occur_in([[item[seq]] for item in base.db[label]]):
                        out += "1 "
                    else:
                        out += "0 "

                out += label
                fout.write(out + "\n")


def generate_arff(name, base, chronicles):
    with open(name, 'w') as fout:
        fout.write('@RELATION features\n\n')
        for i in range(len(chronicles)):
            fout.write('@ATTRIBUTE c' + str(chronicles[i][0].cid) + '\tNUMERIC\n')
        fout.write('@ATTRIBUTE class\t{+,-}\n\n@DATA\n')
        for seq in range(len(base.db_pos[0])):
            out = ""
            for c in chronicles:
                bags = base.get_bags(c[0].get_map_multiset(), db=[[item[seq]] for item in base.db_pos])
                if not c[0].svmkernel:
                    if len(bags) > 0 and c[0].occur(bags[0]):
                        out += "1 "
                    else:
                        out += "0 "
                else:
                    classifier = c[0].classifier
                    if len(bags) > 0:
                        pred = classifier.predict(bags)
                        if np.sign(pred[0]) == "1":
                            out += "1 "
                        else:
                            out += "0 "
                    else:
                        out += "0 "

            out += "+"
            fout.write(out + "\n")
        for seq in range(len(base.db_neg[0])):
            out = ""
            for c in chronicles:
                bags = base.get_bags(c[0].get_map_multiset(), db=[[item[seq]] for item in base.db_neg])
                if not c[0].svmkernel:
                    if len(bags) > 0 and c[0].occur(bags[0]):
                        out += "1 "
                    else:
                        out += "0 "
                else:
                    classifier = c[0].classifier
                    if len(bags) > 0:
                        pred = classifier.predict(bags)
                        if np.sign(pred[0]) == "1":
                            out += "1 "
                        else:
                            out += "0 "
                    else:
                        out += "0 "
            out += "-"
            fout.write(out + "\n")


def mc_predict_label(base, patterns, default='', statistical=False, g=0):
    if not statistical:
        for c in patterns:
            if c[0].occur_in(base):
                return c[1], 1
    else:
        votes = {}
        for c in patterns:
            if c[3] + g > 0:
                if len(votes) > 0:
                    break
                else:
                    if not c[1] in votes:
                        votes[c[1]] = 0
                    votes[c[1]] += float(c[2]) / (float(c[3]) + float(g))
            else:
                if not c[1] in votes:
                    votes[c[1]] = 0
                votes[c[1]] += c[2]
        if len(votes) > 0:
            maxlab = ('', 0)
            for label in votes:
                if votes[label] > maxlab[1]:
                    maxlab = (label, votes[label])
            return maxlab[0], 1

    return default, 0

def classify_sample(patterns, n, g=0):
    if n > 0 and len(patterns) > n:
        return sorted(patterns,
                      key=lambda p: float("inf") if (float(p[3]) + g) == 0 else float(p[2]) / (float(p[3]) + g),
                      reverse=True)[:n]
    return patterns

def mc_pred_with_chronicles(base, patterns, tp=False, clf=None, labels=None, g=0):
    if not clf:
        patterns = sorted(patterns,
                      key=lambda p: float("inf") if (float(p[3]) + g) == 0 else float(p[2]) / (float(p[3]) + g),
                      reverse=True)
    label_predict = {}
    total = [0, 0]
    for label in base.db:
        label_predict[label] = [0, 0]
    label_predict[''] = [0, 0]
    ndef_label_predict = copy.deepcopy(label_predict)
    ndef_total = copy.deepcopy(total)
    if not clf:
        default = ('', 0)
        for label in base.db:
            if len(base.db[label][0]) > default[1]:
                default = (label, len(base.db[label][0]))          
    for label in base.db:
        for seq in range(len(base.db[label][0])):
            if not clf:
                lpredict, is_default = mc_predict_label([[item[seq]] for item in base.db[label]], patterns, default=default[0],
                                        statistical=False, g=g)        
            else:
                lpredict = clf.predict(np.array([features_seq(base, seq, label, patterns)]))[0]
                is_default = 0
                if labels and type(lpredict).__module__ == 'numpy':
                    lpredict = labels[np.argmax(lpredict)]
            
            if lpredict == label:
                label_predict[label][0] += 1
                total[0] += 1
                if is_default == 1:
                    ndef_label_predict[lpredict][0] += 1
                    ndef_total[0] += 1
            else:
                label_predict[label][1] += 1
                total[1] += 1
                if is_default == 1:
                    ndef_label_predict[lpredict][1] += 1
                    ndef_total[1] += 1

    out = "Accuracy: " + str(float(total[0]) / (float(total[0]) + float(total[1]))) + "\n"
    out += "Absolute accuracy: " + str(float(total[0])) + "/" + str(float(total[0]) + float(total[1])) + "\n"
    if ndef_total[0] + ndef_total[1] > 0:
        out += "Accuracy without default prediction: " + str(float(ndef_total[0]) / (float(ndef_total[0]) + float(ndef_total[1]))) + "\n"
        out += "Absolute accuracy without default prediction: " + str(float(ndef_total[0])) + "/" + str(float(ndef_total[0]) + float(ndef_total[1])) + "\n"
    recall = []
    precision = []
    for label in label_predict:
        if label in base.db:
            if len(base.db[label][0]) > 0:
                recall.append(float(label_predict[label][0]) / float(len(base.db[label][0])))
            if label_predict[label][0] + label_predict[label][1] > 0:
                precision.append(
                    float(label_predict[label][0]) / float(label_predict[label][0] + label_predict[label][1]))
    out += "Recall: " + str(np.mean(recall)) + "\n"
    out += "Precision: " + str(np.mean(precision)) + "\n"
    for label in label_predict:
        if label != "" and float(label_predict[label][0]) + float(label_predict[label][1]) > 0:
            out += label + " accuracy: " + str(float(label_predict[label][0])) + "/" + str(float(label_predict[label][0]) + float(label_predict[label][1])) + " (" + str(float(label_predict[label][0]) / (
            float(label_predict[label][0]) + float(label_predict[label][1]))) + ")\n"
        if label != "" and float(ndef_label_predict[label][0]) + float(ndef_label_predict[label][1]) > 0:
            out += label + " accuracy without default prediction: " + str(float(ndef_label_predict[label][0])) + "/" + str(float(ndef_label_predict[label][0]) + float(ndef_label_predict[label][1])) + " (" + str(float(ndef_label_predict[label][0]) / (
            float(ndef_label_predict[label][0]) + float(ndef_label_predict[label][1]))) + ")\n"
            
    if tp:
        with open(str(tp) + "/res_classify", 'w') as fout:
            fout.write(out)
    else:
        print(out)

def pred_with_chronicles(base, patterns, tp=False):
    true_pos = 0
    false_neg = 0
    true_neg = 0
    false_pos = 0
    for seq in range(len(base.db_pos[0])):
        pos = False
        for c in patterns:
            bags = base.get_bags(c[0].get_map_multiset(), db=[[item[seq]] for item in base.db_pos])
            if not c[0].svmkernel:
                if len(bags) > 0 and c[0].occur(bags[0]):
                    pos = True
                    break
            else:
                classifier = c[0].classifier
                if len(bags) > 0:
                    pred = classifier.predict(bags)
                    if np.sign(pred[0]) == "1":
                        pos = True
                        break
        if pos:
            true_pos += 1
        else:
            false_neg += 1
    for seq in range(len(base.db_neg[0])):
        neg = False
        for c in patterns:
            bags = base.get_bags(c[0].get_map_multiset(), db=[[item[seq]] for item in base.db_neg])
            if not c[0].svmkernel:
                if len(bags) > 0 and c[0].occur(bags[0]):
                    neg = True
                    break
            else:
                classifier = c[0].classifier
                if len(bags) > 0:
                    pred = classifier.predict(bags)
                    if np.sign(pred[0]) == "1":
                        neg = True
                        break
        if neg:
            false_pos += 1
        else:
            true_neg += 1

    out = "Accuracy: " + str(float(true_pos + true_neg) / float(false_neg + false_pos + true_pos + true_neg)) + "\n"
    out += "Positive accuracy: " + str(float(true_pos) / float(true_pos + false_neg)) + "\n"
    out += "True positive: " + str(true_pos) + "\n"
    out += "False positive: " + str(false_pos) + "\n"
    out += "True negative: " + str(true_neg) + "\n"
    out += "False negative: " + str(false_neg)

    if tp:
        with open(str(tp) + "/res_classify_none", 'w') as fout:
            fout.write(out)
    else:
        print(out)

def features_seq(train, seq, label, chronicles):
    x = []
    for c in chronicles:
        if c[0].occur_in([[item[seq]] for item in train.db[label]]):
            x.append(1)
        else:
            x.append(0)
    return x        

def fit_svm(train, chronicles, clf):
    X = []
    y = []
    for label in train.db:
        for seq in range(len(train.db[label][0])):
            X.append(features_seq(train, seq, label, chronicles))
            y.append(label)
    clf.fit(np.array(X), np.array(y))

def mc_classify(train, test, patterns, method, kernel="linear", C=1, tp=False, g=0):
    global temp_dir

    if method not in ("none", "svc"):
        classifier = "weka.classifiers.trees.REPTree"
        if method == "nb":
            classifier = "weka.classifiers.bayes.NaiveBayes"
        elif method == "j48":
            classifier = "weka.classifiers.trees.J48"
        elif method == "smo":
            classifier = "weka.classifiers.functions.SMO"
        elif method == "linear":
            classifier = "weka.classifiers.functions.LinearRegression"

        train_feat = tempfile.NamedTemporaryFile(prefix="train_feat_" + method, suffix=".arff", delete=False,
                                                 dir=temp_dir)
        test_feat = tempfile.NamedTemporaryFile(prefix="test_feat_" + method, suffix=".arff", delete=False,
                                                dir=temp_dir)
        mc_generate_arff(train_feat.name, train, patterns)
        mc_generate_arff(test_feat.name, test, patterns)
        train_feat.close()
        test_feat.close()
        cmd = "java -Xmx2024m -cp " + GLOBAL.WEKA + " " + classifier + \
              " -t " + train_feat.name + " -T " + test_feat.name + " -v"
        if tp:
            with open(str(tp) + "/res_classify_" + method, 'wb') as fout:
                subprocess.check_call(cmd, shell=True, stdout=fout)
        else:
            subprocess.check_call(cmd, shell=True)
        os.remove(train_feat.name)
        os.remove(test_feat.name)
    else:
        clf = None
        labels = None
        clf = svm.SVC(C=C, kernel=kernel)
        fit_svm(train, patterns, clf)
        mc_pred_with_chronicles(test, patterns, tp, clf=clf, labels=labels, g=g)


def classify(train, test, patterns, method, tp=False):
    if method != "none":
        classifier = "weka.classifiers.trees.REPTree"
        if method == "nb":
            classifier = "weka.classifiers.bayes.NaiveBayes"
        elif method == "j48":
            classifier = "weka.classifiers.trees.J48"
        elif method == "smo":
            classifier = "weka.classifiers.functions.SMO"
        elif method == "linear":
            classifier = "weka.classifiers.functions.LinearRegression"

        generate_arff("features.arff", train, patterns)
        generate_arff("testfeatures.arff", test, patterns)
        cmd = "java -Xmx2024m -cp " + GLOBAL.WEKA + " " + classifier + \
              " -t features.arff -T testfeatures.arff -v"
        if tp:
            with open(str(tp) + "/res_classify", 'wb') as fout:
                subprocess.check_call(cmd, shell=True, stdout=fout)
        else:
            subprocess.check_call(cmd, shell=True)
    else:
        pred_with_chronicles(test, patterns, tp)


def extract_res_classify(path, suffix="", parse=False):
    reg_classify = ""
    if not parse:
        reg_classify = "^Correctly Classified Instances\s+?[0-9]+?\s+?([0-9\.]+?)\s+?%"
    else:
        reg_classify = "^Accuracy: ([0-9\.]+?)$"

    try:
        with open(path + "/res_classify" + suffix, 'r') as fin:
            for line in fin:
                res = re.match(reg_classify, line)
                if res:
                    if not parse:
                        return float(res.group(1)) / 100.0
                    else:
                        return float(res.group(1))
    except:
        print("[ERROR]" + path + "/res_classify" + suffix, file=sys.stderr)
        return None


def extract_precision(path, suffix="", parse=False):
    reg_classify = "^=== Confusion Matrix ==="
    confused = False
    tp = []
    all = []
    tok = 0

    try:
        with open(path + "/res_classify" + suffix, 'r') as fin:
            for line in fin:
                if not parse:
                    if not confused:
                        res = re.match(reg_classify, line)
                        if res:
                            confused = True
                    else:
                        l = line.replace('\n', '')
                        l = l.split()[:-4]
                        if len(l) > 0 and l[0] != 'a':
                            if len(tp) == 0:
                                tp = [0 for _ in l]
                                all = [0 for _ in l]
                            l = map(int, l)
                            tp[tok] = l[tok]
                            for i in range(len(l)):
                                all[i] += l[i]
                            tok += 1
                else:
                    reg_prec = "^Precision: ([0-9\.]+?)$"
                    res = re.match(reg_prec, line)
                    if res:
                        return float(res.group(1))

        if not parse:
            res = []
            for i in range(len(tp)):
                if tp[i] > 0:
                    res.append(float(tp[i]) / float(all[i]))
            if len(res) > 0:
                return np.mean(res)
        return None
    except:
        return None


def extract_recall(path, suffix="", parse=False, coef=False):
    reg_classify = "^=== Confusion Matrix ==="
    confused = False
    recalls = []
    n = 0

    try:
        with open(path + "/res_classify" + suffix, 'r') as fin:
            for line in fin:
                if not parse:
                    if not confused:
                        res = re.match(reg_classify, line)
                        if res:
                            confused = True
                    else:
                        l = line.replace('\n', '')
                        l = l.split()[:-4]
                        if len(l) > 0 and l[0] != 'a':
                            l = map(int, l)
                            if not coef:
                                recalls.append(float(l[len(recalls)]) / sum(l))
                            else:
                                recalls.append(float(l[len(recalls)]))
                                n += sum(l)
                else:
                    reg_prec = "^Recall: ([0-9\.]+?)$"
                    res = re.match(reg_prec, line)
                    if res:
                        return float(res.group(1))

        if not parse:
            if len(recalls) > 0:
                if not coef:
                    return np.mean(recalls)
                elif n > 0:
                    return float(sum(recalls)) / float(n)
        return None
    except:
        return None
        
def classify_sample(patterns, n, g=0):
    if n > 0 and len(patterns) > n:
        return sorted(patterns,
                      key=lambda p: float("inf") if (float(p[3]) + g) == 0 else float(p[2]) / (float(p[3]) + g),
                      reverse=True)[:n]
    return patterns

def compute_res_classify(path, suffix, my=False, parse=False):
    out = ""
    for dir in os.listdir(path):
        path_dir = path + "/" + dir
        if os.path.isdir(path_dir):
            res_classify = []
            for ex in os.listdir(path_dir):
                path_ex = path_dir + "/" + ex
                if os.path.isdir(path_ex):
                    n = extract_res_classify(path_ex, suffix, parse)
                    if n:
                        res_classify.append(n)
            out += "---------- " + dir + " ----------\n"
            out += "Best accuracy: " + str(np.max(res_classify)) + "\n"
            out += "Worst accuracy: " + str(np.min(res_classify)) + "\n"
            out += "Mean accuracy: " + str(np.mean(res_classify)) + "\n"
            out += "Median accuracy: " + str(np.median(res_classify)) + "\n"
            out += "Standard deviation accuracy: " + str(np.std(res_classify)) + "\n"
    return out
