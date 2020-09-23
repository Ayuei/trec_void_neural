import json

class Parser:
    @classmethod
    def parse_document(cls, *args, **kwargs):
        raise NotImplementedError()


class CovidParser(Parser):
    is_str = lambda k: isinstance(k, str)
    data_path = 'covid-april-10/'

    @classmethod
    def parse(cls, row):
        document = {'id': row['cord_uid'],
                    'title': row['title']}

        if cls.is_str(row['full_text_file']):
            path = None
            if row['has_pmc_xml_parse']:
                path = cls.data_path + row['full_text_file'] + \
                       '/pmc_json/' + row['pmcid'].split(';')[ 0] +\
                       '.xml.json'

            elif row['has_pdf_parse']:
                path = cls.data_path + row['full_text_file'] +\
                       '/pdf_json/' + row['sha'].split(';')[0] +\
                       '.json'

            if path:
                content = json.load(open(path))
                fulltext = '\n'.join([p['text'] for p in content['body_text']])
                document['fulltext'] = fulltext
        else:
            document['fulltext'] = ''

        if cls.is_str(row['abstract']):
            document['abstract']=row['abstract']
        else:
            document['abstract']=''
        if cls.is_str(row['publish_time']):
            date = row['publish_time']
            len_is_4 = len(row['publish_time']) == 4
            document['date'] = f'{date}-01-01' if len_is_4 else date

        return document

class CovidParserNew(CovidParser):
    @classmethod
    def parse(cls, row):
        document = {'id': row['cord_uid'],
                    'title': row['title']}

        path = None
        if cls.is_str(row['pdf_json_files']):
            data_row = row['pdf_json_files'].split(';')[0].strip()
            path = cls.data_path + data_row

        elif cls.is_str(row['pmc_json_files']):
            data_row = row['pmc_json_files'].split(';')[0].strip()
            path = cls.data_path + data_row

        if path:
            content = json.load(open(path))
            fulltext = '\n'.join([p['text'] for p in content['body_text']])
            document['fulltext'] = fulltext

        else:
            document['fulltext'] = ''

        if cls.is_str(row['abstract']):
            document['abstract']=row['abstract']
        else:
            document['abstract']=''
        if cls.is_str(row['publish_time']):
            date = row['publish_time']
            len_is_4 = len(row['publish_time']) == 4
            document['date'] = f'{date}-01-01' if len_is_4 else date

        return document
