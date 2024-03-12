import codecs


def save_results(input_list, output_path):
    with codecs.open(output_path, 'w', encoding='utf-8') as fout:
        for ll in input_list:
            line = '\t'.join(['%.4f' % v for v in ll]) + '\n'
            fout.write(line)


def save_train_config(args, n_train, n_val, patience, best_epochs):
    with codecs.open(args.wrirte_config_path, 'w', encoding='utf-8') as f:
        line = f'net:' + '\t' + f'{args.net}' + '\n'
        f.write(line)
        line = f'epochs:' + '\t' + f'{args.epochs}' + '\n'
        f.write(line)
        line = f'batch_size:' + '\t' + f'{args.batchsize}' + '\n'
        f.write(line)
        line = f'scale:' + '\t' + f'{args.scale}' + '\n'
        f.write(line)
        line = f'Loss:' + '\t' + f'{args.loss}' + '\n'
        f.write(line)
        line = f'Train size:' + '\t' + f'{n_train}' + '\n'
        f.write(line)
        line = f'Val size:' + '\t' + f'{n_val}' + '\n'
        f.write(line)
        line = f'Deep Supervision:' + '\t' + f'{args.DS}' + '\n'
        f.write(line)
        line = f'Early stopping patience:' + '\t' + f'{patience}' + '\n'
        f.write(line)
        line = f'Best epochs:' + '\t' + f'{best_epochs}' + '\n'
        f.write(line)
        line = f'seed:' + '\t' + f'{args.seed}' + '\n'
        f.write(line)
        line = f'learning rate:' + '\t' + f'{args.lr}' + '\n'
        f.write(line)


def save_test_result(test_result_path, totall_result):
    with codecs.open(test_result_path, 'w', encoding='utf-8') as f:
        line = f'epoch' + '\t' + f'Dice' + '\t' + f'Iou' + '\t' + f'Recall' + '\t' + f'Precision' + '\t' + f'F1_score' + '\t' + f'test_loss' + '\t' + \
               f'Accuracy' + '\t' + f'specificity' + '\t' + f'sensitivity' + '\t' + f'Auc' + '\t' + f'Ap' + '\n'
        f.write(line)
        for i, ll in enumerate(totall_result):
            line = f'%d' % (i + 1) + '\t' + '%.4f' % ll['dice'].avg + '\t' + '%.4f' % ll[
                'iou'].avg + '\t' + '%.4f' % ll['recall'].avg + '\t' \
                   + '%.4f' % ll['precision'].avg + '\t' + '%.4f' % ll['f1_score'].avg + '\t' + \
                   '%.4f' % ll['test_loss'].avg + '\t' + '%.4f' % ll['Accuracy'].avg + '\t' + \
                   '%.4f' % ll['specificity'].avg + '\t' + '%.4f' % ll['sensitivity'].avg + '\t' + \
                   '%.4f' % ll['Auc'].avg + '\t' + '%.4f' % ll['Ap'].avg + '\n'
            f.write(line)


def save_val_result(val_result_path, totall_result, lr_list):
    with codecs.open(val_result_path, 'w', encoding='utf-8') as f:
        line = f'epoch' + '\t' + f'Dice' + '\t' + f'Iou' + '\t' + f'Recall' + '\t' + f'Precision' + '\t' + f'F1_score' + '\t' + f'val_loss' + '\t' + \
               f'Accuracy' + '\t' + f'specificity' + '\t' + f'sensitivity' + '\t' + f'Auc' + '\t' + f'Ap' + '\t' + f'lr' + '\n'
        f.write(line)
        for i, ll in enumerate(totall_result):
            line = f'%d' % (i + 1) + '\t' + '%.4f' % ll['dice'].avg + '\t' + '%.4f' % ll[
                'iou'].avg + '\t' + '%.4f' % ll['recall'].avg + '\t' \
                   + '%.4f' % ll['precision'].avg + '\t' + '%.4f' % ll['f1_score'].avg + '\t' + \
                   '%.4f' % ll['val_loss'].avg + '\t' + '%.4f' % ll['Accuracy'].avg + '\t' + \
                   '%.4f' % ll['specificity'].avg + '\t' + '%.4f' % ll['sensitivity'].avg + '\t' + \
                   '%.4f' % ll['Auc'].avg + '\t' + '%.4f' % ll['Ap'].avg + '\t' + '%.8f' % lr_list[i] + '\n'
            f.write(line)


def save_train_result(train_result_path, totall_result, lr_list):
    with codecs.open(train_result_path, 'w', encoding='utf-8') as f:
        line = f'epoch' + '\t' + f'Dice' + '\t' + f'Iou' + '\t' + f'avg_loss' + '\n'
        f.write(line)
        for i, ll in enumerate(totall_result):
            line = f'%d' % (i + 1) + '\t' + '\t'.join(['%.4f' % v for v in ll]) + '\t' + '%.4f' % lr_list[
                i] + '\n'
            f.write(line)
