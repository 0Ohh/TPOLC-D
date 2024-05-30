import ioh


problem = ioh.get_problem(
    'MaxCut2000',
    instance=1,
    dimension=1,
    problem_class=ioh.ProblemClass.GRAPH
)

def get_n():
    return problem.bounds.lb.shape[0]

def FS(x):
    return problem(x)

