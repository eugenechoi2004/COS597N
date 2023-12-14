from Bio import SeqIO

def subset_fasta(input_file, output_file, max_sequences):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for i, record in enumerate(SeqIO.parse(infile, "fasta")):
            if i >= max_sequences:
                break
            SeqIO.write(record, outfile, "fasta")

input_fasta = "../data/raw_data/2mil.fasta"  # Path to your original FASTA file
output_fasta = "../data/raw_data/100k.fasta"  # Path to the output FASTA file
max_sequences = 100_000  # Number of sequences to keep

subset_fasta(input_fasta, output_fasta, max_sequences)