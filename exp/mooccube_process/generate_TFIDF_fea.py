import math

datasets = ["course","cit","mit","prc","cmu"]
for dataset in datasets:
    # concept:count
    concept_dic = {}
    course_count = 0
    with open("data/" + dataset + ".lsvm", "r") as f:
        lines = f.readlines();
        for line in lines:
            course_count += 1
            split_line_list = line.split(" ")[1:]
            for item in split_line_list:
                item = item.split(":")[0]
                if item not in concept_dic.keys():
                    concept_dic[item] = 1
                else:
                    concept_dic[item] += 1
    print(len(concept_dic))

    num_of_documents = course_count

    with open("data/" + dataset + ".lsvm", "r") as f1:
        with open("data/" + dataset + "-TFIDF.lsvm", "w") as f2:
            lines = f1.readlines();
            for line in lines:
                split_line_list = line.split(" ")
                pre_item = split_line_list[0]
                fea_items = split_line_list[1:]
                write_line = "" + pre_item
                for fea in fea_items:
                    if fea == '\n':
                        break
                    item, bow = fea.split(":")
                    weight = round(int(bow) * math.log(num_of_documents / concept_dic[item], 10))
                    line = str(item)
                    write_line += " %d:%d" % (int(item), weight)
                write_line += '\n'
                f2.write(write_line)
        f2.close()
    f1.close()
    print(dataset+":done")
