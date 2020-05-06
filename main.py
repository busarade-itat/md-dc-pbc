#!/usr/bin/env python
import os
import shutil
import sys
import tempfile
import datetime
import re
import subprocess
import GLOBAL
from new_file_format_converter import NewFileFormatConverter

from vectors import DomainElementFactory

from argparse import ArgumentParser

from functions_p3 import McDataset
from functions_p3 import Chronicle
from functions_p3 import mc_classify
from functions_p3 import classify_sample

temp_dir = "temp"


def use_cpp_chro(p, label, fmin, gmin, mincs, maxcs, train_base,
                 episode=False, vecsize=1, debug_mockfiles=None, fold_index=None):
    if debug_mockfiles is None:
        inputposfile = tempfile.NamedTemporaryFile(prefix="input_pos", delete=False, dir=temp_dir)
        inputposfile.write(bytes(train_base.write_seq(label), 'utf_8'))
        inputposfile.close()

        inputnegfile = tempfile.NamedTemporaryFile(prefix="input_neg", delete=False, dir=temp_dir)
        for l in train_base.db_seq:
            if l != label:
                inputnegfile.write(bytes(train_base.write_seq(l), 'utf_8'))
        inputnegfile.close()

        outputfile = tempfile.NamedTemporaryFile(prefix="output", delete=False, dir=temp_dir)
        outputfile.close()

        extractor = GLOBAL.CPP_DCM
        cmd = " ".join((
            extractor,
            inputposfile.name,
            str(fmin),
            '-d', inputnegfile.name,
            '-g', str(gmin),
            '--mincs', str(mincs),
            '--maxcs', str(maxcs),
            '--vecsize', str(vecsize),
            '--verbose' if p else ''
        ))
        if episode:
            cmd += " -e"
        if p:
            print("[INFO] " + cmd, file=sys.stderr)

        subprocess.check_call(cmd, shell=True, stdout=open(outputfile.name, 'wb'), stderr=sys.stderr.buffer)
        os.remove(inputposfile.name)
        os.remove(inputnegfile.name)

        res = Chronicle.parse_cpp(outputfile.name, train_base.reverse_item_mapping)
        os.remove(outputfile.name)

        for c in res:
            c[1] = label
            c[0].set_parent_dataset(train_base)

        return res
    else:
        mock_input_filename = debug_mockfiles[fold_index*2 + (0 if label == "neg" else 1)]

        res = Chronicle.parse_cpp(mock_input_filename, train_base.reverse_item_mapping)

        for c in res:
            c[1] = label
            c[0].set_parent_dataset(train_base)

        return res


def main():
    global temp_dir

    pars = ArgumentParser()
    pars.add_argument("datasets")

    pars.add_argument("--fmin", dest="fmin", type=float, default=0.2)
    pars.add_argument("--gmin", dest="gmin", type=float, default=2)

    pars.add_argument("--mincs", dest="mincs", type=int, default=2)
    pars.add_argument("--maxcs", dest="maxcs", type=int, default=-1)
    pars.add_argument("--test", dest="test", type=float, default=0.2)
    pars.add_argument("--fold", dest="fold", type=int, default=5)
    pars.add_argument("--k", dest="k", type=float, default=0)
    pars.add_argument("--n", dest="n", type=int, default=0)
    pars.add_argument("--C", dest="C", type=float, default=1)
    pars.add_argument("--kernel", dest="kernel", type=str, default="linear")
    pars.add_argument("--verbose", dest="verbose", action="store_true")
    pars.add_argument("--classifier", dest="classifier", type=str, default="svc")

    pars.add_argument("--vecsize", dest="vecsize", type=int, default=1)
    pars.add_argument("-s", dest="vecsize", type=int, default=1)

    pars.add_argument("--legacy", dest="legacy", action="store_true")
    pars.add_argument("-l", dest="legacy", action="store_true")

    pars.add_argument("--component-debug-in", dest="debug_mockfiles", type=str, default=None)
    pars.add_argument("--out", dest="debug_outdir", type=str, default=None)
    pars.add_argument("--convert-only", dest="convert_only", action="store_true")
    pars.add_argument("--disable-randomness", dest="disable_randomness", action="store_true")

    args = pars.parse_args(sys.argv[1:])

    if not args.convert_only:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        if args.debug_outdir is None:
            trace_pattern = "xp"+str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")
        else:
            trace_pattern = args.debug_outdir

        os.mkdir(trace_pattern)
        with open(trace_pattern+"/cmd", 'w') as fout:
            fout.write(" ".join(sys.argv))

    if args.debug_mockfiles is not None:
        setattr(args, "debug_mockfiles", args.debug_mockfiles.split(","))

        if len(args.debug_mockfiles) % 2 != 0:
            raise Exception("Number of mock files must be even.")

        setattr(args, "k", len(args.debug_mockfiles)/2)

    if not args.legacy:
        if args.verbose:
            print("[INFO] New file format conversion...", file=sys.stderr)

        if args.convert_only:
            setattr(args, "datasets", NewFileFormatConverter.CONVERTONLY_OUTPUT_DIR)

            if not os.path.exists(NewFileFormatConverter.CONVERTONLY_OUTPUT_DIR):
                os.makedirs(NewFileFormatConverter.CONVERTONLY_OUTPUT_DIR)

            conv = NewFileFormatConverter(
                args.datasets,
                conversion_out_dir=NewFileFormatConverter.CONVERTONLY_OUTPUT_DIR
            )
            conv.convert_new_to_old()
        else:
            conv = NewFileFormatConverter(
                args.datasets,
                conversion_out_dir=NewFileFormatConverter.CONVERSION_OUTPUT_DIR
            )
            vecsize = conv.convert_new_to_old()

            setattr(args, "vecsize", vecsize)
            setattr(args, "datasets", NewFileFormatConverter.CONVERSION_OUTPUT_DIR)

        if args.verbose or args.convert_only:
            print("[INFO] Converted into " + args.datasets + " with vecsize=" + str(args.vecsize) + ".", file=sys.stderr)

        if args.convert_only:
            exit(0)

    DomainElementFactory.VECTOR_SIZE = args.vecsize

    base = McDataset()
    reg_name = "(.*?)\.dat"
    for file in os.listdir(args.datasets):
        res = re.match(reg_name, file)
        base.load_db_line(args.datasets+"/"+file, res.group(1), store_seq=True)

    fmin = 2
    if float(args.fmin) > 1:
        fmin = int(int(args.fmin)*(1-args.test))
    gmin = float(args.gmin)

    res_classify = {}
    for step in range(0, args.fold):
        if args.verbose:
            print("[INFO] Fold "+str(step), file=sys.stderr)

        if args.disable_randomness is None:
            base.shuffle()

        train_base = McDataset(base.item_mapping)
        train_base.max_occ = base.max_occ
        test_base = McDataset(base.item_mapping)
        test_base.max_occ = base.max_occ
        train_base.set_reverse_item_mapping(base.reverse_item_mapping)
        test_base.set_reverse_item_mapping(base.reverse_item_mapping)

        for label in base.db:
            threshold = int(len(base.db[label][0])*args.test)
            train_base.db[label] = [item[threshold:] for item in base.db[label]]
            train_base.db_seq[label] = base.db_seq[label][threshold:]
            test_base.db[label] = [item[:threshold] for item in base.db[label]]
            test_base.db_seq[label] = base.db_seq[label][:threshold]

        all_discr = []

        for label in train_base.db:
            if float(args.fmin) < 1:
                fmin = int(float(args.fmin)*float(len(train_base.db[label][0])))
            all_discr += classify_sample(
                use_cpp_chro(
                    args.verbose,
                    label,
                    fmin,
                    gmin,
                    args.mincs,
                    args.maxcs,
                    train_base,
                    episode=False,
                    vecsize=args.vecsize,
                    debug_mockfiles=args.debug_mockfiles,
                    fold_index=step
                ),
                args.n,
                g=args.k
            )
        if args.verbose:
            print("[INFO] Test prediction", file=sys.stderr)

        tp_path = str(trace_pattern)+"/ex"+str(step)
        os.mkdir(tp_path)

        with open(tp_path+"/chronicles", 'w') as fout:
            for c, l, g0, g1 in all_discr:
                fout.write(str(c) + "class: "+l+"\nsup(c,pos)/sup(c,neg): " + str(g0)+"/"+str(g1) + "\n\n")
        mc_classify(train_base, test_base, all_discr, args.classifier, kernel=args.kernel, C=args.C, tp=tp_path, g=args.k)

    if not args.convert_only and os.path.exists(temp_dir):
        if not args.legacy:
            os.remove(
                NewFileFormatConverter.CONVERSION_OUTPUT_DIR + "/" +
                NewFileFormatConverter.LABEL_OUTPUT_FILES["0"]
            )
            os.remove(
                NewFileFormatConverter.CONVERSION_OUTPUT_DIR + "/" +
                NewFileFormatConverter.LABEL_OUTPUT_FILES["1"]
            )
            os.rmdir(NewFileFormatConverter.CONVERSION_OUTPUT_DIR)

        shutil.rmtree(temp_dir)
                
if __name__ == "__main__":
    main()

