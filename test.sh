python3 main.py --log results/base-adult.txt 
python3 main.py --log results/w-adult.txt --ite

python3 main.py --log results/base-crime.txt --data communities.data --batch_size 20
python3 main.py --log results/w-crime.txt --data communities.data --batch_size 20 --ite

python3 main.py --log results/base-german.txt --data german.data --batch_size 20
python3 main.py --log results/w-german.txt --data german.data --batch_size 20 --ite

python3 main.py --log results/base-bank.txt --data bank.csv --batch_size 100
python3 main.py --log results/w-bank.txt --data bank.csv --batch_size 100 --ite


