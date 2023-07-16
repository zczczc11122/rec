import logging
logger = logging.getLogger('template_video_classification')

def parse_label_str(lowest_level_classes, sep, is_padding=True, max_level=None, lowest_level_class_ids=None):
    if lowest_level_class_ids is None:
        lowest_level_class_ids = list(range(len(lowest_level_classes)))
    assert len(lowest_level_classes) == len(lowest_level_class_ids), "label 与 id 长度不相等"
    max_level_tmp = max([len(i.split(sep)) for i in lowest_level_classes])
    if max_level is not None and is_padding:
        assert max_level_tmp <= max_level, "max_level设置太小，存在标签等级超过max_level"
    else:
        max_level = max_level_tmp
    hierarchical_label = {}
    for cls_index, label_str in enumerate(lowest_level_classes):
        if label_str == '':
            raise ValueError("标签不能为空字符串")
        labels = label_str.split(sep)
        if is_padding:
            for level in range(max_level):
                temp = sep.join(labels[0:level + 1])
                if level not in hierarchical_label:
                    hierarchical_label[level] = {'cls2id': {}, 'id2cls':{}}
                if temp not in hierarchical_label[level]['cls2id']:
                    if temp == label_str and level == (max_level - 1):
                        cls_id = lowest_level_class_ids[cls_index]
                    else:
                        cls_id = len(hierarchical_label[level]['cls2id'])
                    hierarchical_label[level]['cls2id'][temp] = cls_id
                    hierarchical_label[level]['id2cls'][cls_id] = temp
        else:
            for level in range(max_level):
                if len(labels) <= level:
                    break
                temp = sep.join(labels[0:level + 1])
                if level not in hierarchical_label:
                    hierarchical_label[level] = {'cls2id': {}, 'id2cls': {}}
                if temp not in hierarchical_label[level]['cls2id']:
                    cls_id = len(hierarchical_label[level]['cls2id'])
                    hierarchical_label[level]['cls2id'][temp] = cls_id
                    hierarchical_label[level]['id2cls'][cls_id] = temp
    return hierarchical_label

def parse_label_file(filename, sep="|", is_padding=True, max_level=None):
    class_ids = []
    class_labels = []
    with open(filename) as fh:
        for line in fh:
            line_split = line.strip().split(' ')
            class_id = int(line_split[0])
            class_label = line_split[1]
            class_ids.append(class_id)
            class_labels.append(class_label)
    label_info = parse_label_str(class_labels, sep=sep, is_padding=is_padding, max_level=max_level, lowest_level_class_ids=class_ids)
    return label_info

def parse_label(args, dims):
    file_expression = args.idx2class_file_expression
    file_material = args.idx2class_file_material
    file_person = args.idx2class_file_person
    file_style = args.idx2class_file_style
    file_topic = args.idx2class_file_topic

    expression_label_info = parse_label_file(file_expression)
    material_label_info = parse_label_file(file_material)
    person_label_info = parse_label_file(file_person)
    style_label_info = parse_label_file(file_style)
    topic_label_info = parse_label_file(file_topic)

    label_dict = {}

    for dim in dims:
        label_info = eval(dim + '_label_info')
        for level in sorted(label_info.keys()):
            logger.info('Dataset {} level{} Label: {}'.
                        format(dim, level, ' '.join(['{}_{}'.format(k, v) for k, v in label_info[level]['id2cls'].items()])))
        logger.info("====================")
        label_dict[dim] = label_info
    return label_dict

