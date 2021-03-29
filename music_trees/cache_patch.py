from os import write
import shelve
from pathlib import Path

train_shelf_path = Path('cache') / 'mdb' / 'train' / 'cache.pag'
test_shelf_path = Path('cache') / 'mdb' / 'test' / 'cache.pag'
output_shelf_path = Path('cache') / 'mdb' / 'logmel-win512-hop128.pag'

with shelve.open(str(output_shelf_path), writeback=True) as out_shelf:
    with shelve.open(str(train_shelf_path), writeback=False) as train_shelf:
        out_shelf.update(train_shelf)
    with shelve.open(str(test_shelf_path), writeback=False) as test_shelf:
        out_shelf.update(test_shelf)
    
    out_shelf.sync()



