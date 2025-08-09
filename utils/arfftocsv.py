
def parse_attributes(arff_lines):
    attributes = []
    for line in arff_lines:
        line = line.strip()
        if line.startswith('@attribute'):
            parts = line.split()
            attributes.append(parts[1])
        elif line.startswith('@data'):
            break
    return attributes


def parse_data(arff_lines, attributes):
    data = []
    in_data_section = False
    for line in arff_lines:
        line = line.strip()
        if line.startswith('@data'):
            in_data_section = True
            continue
        if in_data_section and line:
            row = [0] * len(attributes)
            sparse_pairs = line[1:-1].split(',')
            for pair in sparse_pairs:
                idx, value = pair.split()
                idx = int(idx)
                row[idx] = int(value)
            data.append(row)
    return data



def write_to_csv(attributes, data, csv_file_path):
    with open(csv_file_path, 'w', encoding='utf-8') as csv_file:
        csv_file.write(','.join(attributes) + '\n')
        for row in data:
            csv_file.write(','.join(str(x) for x in row) + '\n')


def arff_to_csv(arff_file_path, csv_file_path):
    with open(arff_file_path, 'r', encoding='utf-8') as arff_file:
        arff_lines = arff_file.readlines()

    attributes = parse_attributes(arff_lines)
    data = parse_data(arff_lines, attributes)
    write_to_csv(attributes, data, csv_file_path)


if __name__ == "__main__":
    arff_file_path = 'delicious.arff' 
    csv_file_path = 'delicious.csv'  
    arff_to_csv(arff_file_path, csv_file_path)