import logging
logger = logging.getLogger('template_video_classification')

def parse_label(args):
    idx2class_person = {}
    class2idx_person = {}
    with open(args.idx2class_file_person) as fh:
        for line in fh:
            line_split = line.strip().split(' ')
            class_id = int(line_split[0])
            class_label = line_split[1]
            idx2class_person[class_id] = class_label
            class2idx_person[class_label] = class_id
    args.num_class_person = len(idx2class_person)
    logger.info('Dataset Person Label: {}'.format(' '.join(['{}_{}'.format(k, v) for k, v in idx2class_person.items()])))

    idx2class_expression = {}
    class2idx_expression = {}
    with open(args.idx2class_file_expression) as fh:
        for line in fh:
            line_split = line.strip().split(' ')
            class_id = int(line_split[0])
            class_label = line_split[1]
            idx2class_expression[class_id] = class_label
            class2idx_expression[class_label] = class_id
    args.num_class_expression = len(idx2class_expression)
    logger.info(
        'Dataset expression Label: {}'.format(' '.join(['{}_{}'.format(k, v) for k, v in idx2class_expression.items()])))

    idx2class_style = {}
    class2idx_style = {}
    with open(args.idx2class_file_style) as fh:
        for line in fh:
            line_split = line.strip().split(' ')
            class_id = int(line_split[0])
            class_label = line_split[1]
            idx2class_style[class_id] = class_label
            class2idx_style[class_label] = class_id
    args.num_class_style = len(idx2class_style)
    logger.info(
        'Dataset style Label: {}'.format(' '.join(['{}_{}'.format(k, v) for k, v in idx2class_style.items()])))

    idx2class_topic = {}
    class2idx_topic = {}
    with open(args.idx2class_file_topic) as fh:
        for line in fh:
            line_split = line.strip().split(' ')
            class_id = int(line_split[0])
            class_label = line_split[1]
            idx2class_topic[class_id] = class_label
            class2idx_topic[class_label] = class_id
    args.num_class_topic = len(idx2class_topic)
    logger.info(
        'Dataset topic Label: {}'.format(' '.join(['{}_{}'.format(k, v) for k, v in idx2class_topic.items()])))

    label_dict = {}

    label_dict["person"] = {}
    label_dict["person"]["id2cls"] = idx2class_person
    label_dict["person"]["cls2id"] = class2idx_person

    label_dict["expression"] = {}
    label_dict["expression"]["id2cls"] = idx2class_expression
    label_dict["expression"]["cls2id"] = class2idx_expression

    label_dict["style"] = {}
    label_dict["style"]["id2cls"] = idx2class_style
    label_dict["style"]["cls2id"] = class2idx_style

    label_dict["topic"] = {}
    label_dict["topic"]["id2cls"] = idx2class_topic
    label_dict["topic"]["cls2id"] = class2idx_topic
    return label_dict
