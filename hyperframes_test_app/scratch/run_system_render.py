import subprocess

p = subprocess.Popen(
    ['node', 'run-storyboard.js'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd='c:\\Users\\home\\Desktop\\CG-DEV\\CD-APP1FB\\hyperframes_test_app'
)

# Send inputs: [Path to system storyboard, Option 2 to render]
stdout, stderr = p.communicate(input="outputs/vl_6108ab3e/lesson.json\n2\n")
print("STDOUT:")
print(stdout)
print("STDERR:")
print(stderr)
