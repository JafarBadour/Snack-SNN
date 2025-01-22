for ((i=50; i<=100; i+=5)); do
    echo Experiement Sparsity $i dense then sparse multiplication
    python test_sparse_multiply.py a $i
    python test_sparse_multiply.py b $i
done