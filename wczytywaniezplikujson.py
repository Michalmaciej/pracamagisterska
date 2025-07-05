import json

counter = 0

def reader(counter):
    file_path = 'WLASL_v0.3.json'
    output_file = 'output.txt'
    train_file = 'train.txt'
    test_file = 'test.txt'
    validate_file = 'validate.txt'

    with open(file_path) as ipf:
        content = json.load(ipf)

    with open(output_file, 'w') as out_f, open(train_file, 'w') as train_f, open(test_file, 'w') as test_f, open(validate_file, 'w') as val_f:
        header = "gloss\tsplit\tvideo_id\tsigner_id\tvariation_id\n"
        out_f.write(header)
        train_f.write(header)
        test_f.write(header)
        val_f.write(header)

        selected_glosses = ["book", "computer", "before", "drink"]  

        for ent in content:
            counter=0
            gloss = ent['gloss']
            if gloss not in selected_glosses:
                continue

            for inst in ent['instances']:
                split = inst['split']
                video_id = inst['video_id']
                signer_id = inst['signer_id']
                variation_id = inst['variation_id']

                line = f"{gloss}\t{split}\t{video_id}\t{signer_id}\t{variation_id}\n"
                out_f.write(line)

                if split == "train":
                    train_f.write(line)
                elif split == "test":
                    test_f.write(line)
                elif split == "val":
                    val_f.write(line)


    return counter

print('total glosses:', reader(counter))
