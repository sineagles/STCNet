import xlsxwriter


def write_PR(p, r, t, f1, filename):
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    worksheet.activate()
    title = ['t', 'precision', 'recall', 'f1score']
    worksheet.write_row('A1', title)

    n_row = 2
    for i in range(len(p)):
        insertData = [t[i], p[i], r[i], f1[i]]
        row = 'A' + str(n_row)
        worksheet.write_row(row, insertData)
        n_row = n_row + 1

    workbook.close()
