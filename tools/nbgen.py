import nbformat as nbf
class NB:
    def __init__(self): self.nb=nbf.v4.new_notebook(); self.nb.metadata['kernelspec']={'name':'cardml','display_name':'Python (cardml)','language':'python'}
    def md(self,t): self.nb.cells.append(nbf.v4.new_markdown_cell(t.strip('\n')))
    def code(self,t): self.nb.cells.append(nbf.v4.new_code_cell(t.strip('\n')))
    def save(self,p): nbf.write(self.nb,p); print(p,len(self.nb.cells),'cells')
