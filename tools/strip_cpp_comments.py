"""Remove C/C++ comments while preserving literals and line boundaries.

Used to prepare the vendored training examples, not during a normal build.
"""
import re


def strip_comments(source):
    out = []
    i = 0
    while i < len(source):
        # Raw strings can contain quotes, slashes and newlines. Prefixes such
        # as u8R are copied normally until we reach the R.
        raw = re.match(r'R"([^ ()\\\t\r\n]{0,16})\(', source[i:]) if source[i] == 'R' else None
        if raw:
            end_marker = ')' + raw.group(1) + '"'
            end = source.find(end_marker, i + raw.end())
            if end < 0:
                raise ValueError('unterminated raw string')
            end += len(end_marker)
            out.append(source[i:end])
            i = end
        elif (source[i] == "'" and i + 1 < len(source)
              and source[i + 1].isalnum()
              and re.search(r"\b(?:0[xX][0-9a-fA-F']*|"
                            r"0[bB][01']*|[0-9][0-9a-zA-Z.']*)$",
                            source[max(0, i - 128):i])):
            # C++ digit separator, not the start of a character literal.
            out.append(source[i])
            i += 1
        elif source[i] in "\"'":
            quote = source[i]
            start = i
            i += 1
            while i < len(source):
                if source[i] == '\\':
                    i += 2
                elif source[i] == quote:
                    i += 1
                    break
                else:
                    i += 1
            else:
                raise ValueError('unterminated string or character literal')
            out.append(source[start:i])
        elif source.startswith('//', i):
            i += 2
            out.append(' ')
            while i < len(source):
                if source[i] == '\\' and source[i + 1:i + 2] == '\n':
                    out.append('\n')
                    i += 2
                elif source[i] == '\n':
                    break
                else:
                    i += 1
        elif source.startswith('/*', i):
            end = source.find('*/', i + 2)
            if end < 0:
                raise ValueError('unterminated block comment')
            out.append(' ' + '\n' * source[i:end + 2].count('\n'))
            i = end + 2
        else:
            out.append(source[i])
            i += 1
    return ''.join(out)


def prepare_source(source):
    """Normalize upstream indentation and excess empty lines for training.

    The vendored inputs have no multiline raw string literals; blank-line
    compaction is therefore only applied after verifying this property.
    """
    source = strip_comments(source).replace('\t', '    ').replace('\f', '\n')
    # Do not silently change the contents of a multiline raw literal.
    for match in re.finditer(r'R"([^ ()\\\t\r\n]{0,16})\(', source):
        end = source.find(')' + match.group(1) + '"', match.end())
        if end < 0 or '\n' in source[match.end():end]:
            raise ValueError('multiline or unterminated raw literal')
    lines = []
    blanks = 0
    for line in source.splitlines():
        if not line.strip():
            blanks += 1
            if blanks <= 2:
                lines.append('')
        else:
            blanks = 0
            lines.append(line)
    return '\n'.join(lines).strip('\n') + '\n'
