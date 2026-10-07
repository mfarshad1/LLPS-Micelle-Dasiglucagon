#!/bin/bash
set -euo pipefail

ROOT=$(pwd -P)

if [ ! -f "$ROOT/in.genQ" ]; then
    echo "ERROR: parent template in.genQ not found in $ROOT"
    exit 1
fi

if [ ! -f "$ROOT/system.data" ]; then
    echo "ERROR: system.data not found in $ROOT"
    exit 1
fi

if ! grep -q "RSEED" "$ROOT/in.genQ"; then
    echo "ERROR: in.genQ does not contain RSEED"
    echo "In parent in.genQ, the Langevin line should look like:"
    echo "fix 1 all langevin 1.0 1.0 1.0 RSEED"
    exit 1
fi

for dir in q-*
do
    [ -d "$dir" ] || continue

    qname=${dir#q-}
    qint=${qname%.*}

    for rep in 1 2 3
    do
        repdir="$ROOT/$dir/rep$rep"
        mkdir -p "$repdir"

        seed=$((1000000 + 10000*rep + 100*qint + 29))

        echo "Preparing $dir replica $rep with seed $seed"

        sed \
            -e "s/QQQQQ/$qname/g" \
            -e "s/RSEED/$seed/g" \
            -e "s|\.\./system.data|../../system.data|g" \
            "$ROOT/in.genQ" > "$repdir/in.genQ.$qname"

        cat > "$repdir/sub.genQ.$qname.rep$rep" <<EOF
#!/bin/bash

#$ -M mfarshad@nd.edu
#$ -m abe
#$ -pe smp 12
#$ -q long
#$ -N M+DG-q-$qname-r$rep

cd $repdir

module load lammps

mpirun -np \$NSLOTS lmp_mpi < in.genQ.$qname
EOF

        cd "$repdir"
        rm -f *.o*
        qsub "sub.genQ.$qname.rep$rep"
        cd "$ROOT"
    done
done
