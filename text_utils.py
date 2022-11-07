import stanza

nlp = stanza.Pipeline(lang='id', processors='tokenize')


def get_sentence_tokens(sent, lower=True):
    doc = nlp(sent)
    if lower:
        return [token.text.lower() for sentence in doc.sentences for token in sentence.tokens]
    else:
        return [token.text for sentence in doc.sentences for token in sentence.tokens]


def get_sentences_tokens(text):
    doc = nlp(text)
    tokens, sentences = [], []
    for sentence in doc.sentences:
        sent_tok = []
        for token in sentence.tokens:
            sent_tok.append(token.text)
        tokens.append(sent_tok)
        sentences.append(sentence.text)
    return tokens, sentences


def html_decode(s):
    """
    Returns the ASCII decoded version of the given HTML string. This does
    NOT remove normal HTML tags like <p>.
    """
    htmlCodes = (
            ("'", '&#39;'),
            ('"', '&quot;'),
            ('>', '&gt;'),
            ('<', '&lt;'),
            ('&', '&amp;'),
            ("'", '&#39'),
            ('"', '&quot'),
            ('>', '&gt'),
            ('<', '&lt'),
            ('&', '&amp')
        )
    for code in htmlCodes:
        s = s.replace(code[1], code[0])
    return s
