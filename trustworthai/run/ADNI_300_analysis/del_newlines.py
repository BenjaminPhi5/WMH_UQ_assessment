prev_line = '\n'
kept_lines = []
with open("combined_logs.txt", "r") as f:
    for line in f.readlines():
        line_nospace = line.replace(" ", "")
        if prev_line !='\n' or (line_nospace != '\n' and '' not in line):
            kept_lines.append(line)
        prev_line = line_nospace
        
with open("modded_combined_logs.txt", "w") as f:
    for line in kept_lines:
        f.write(line)