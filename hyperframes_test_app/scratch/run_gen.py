import subprocess

p = subprocess.Popen(
    ['python', 'run_hyperframes_helper.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd='c:\\Users\\home\\Desktop\\CG-DEV\\CD-APP1FB'
)

# Send inputs: [Choice 2, Topic: Human Digestive System, Theme: Science]
stdout, stderr = p.communicate(input="2\nHuman Digestive System\nScience\n")
print("STDOUT:")
print(stdout)
print("STDERR:")
print(stderr)
