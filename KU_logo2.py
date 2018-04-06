import random
from PIL import Image,ImageDraw
import copy
class Polygon:
    def __init__(self,minsize,maxsize,w,h,corner,edges = 4):
        assert(maxsize >= minsize)
        self.edges = edges
        self.area = []
        #self.content = []
        self.length = random.randint(minsize,maxsize)
        self.corner = corner
        self.error = 0
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        self.color = [r,g,b]

    img = Image.open('test1.png')
    h = img.size[1]
    w = img.size[0]
    data = img.getdata()
    diff = 0
    def compute_maxdiff():
        global diff
        for x in range(w):
            for y in range(h):
                c = data[y*w+x]
                diff += sum(max(255-c[i],c[i]) for i in range(3))
        print(diff)
    def compute(c1,c2):
        return sum(abs(c1[i]-c2[i]) for i in range(3))
class Picture:
    def __init__(self,w,h):
        self.w = w
        self.h = h
        self.polygons = []
        self.error = 0

    def add(self,polygon):
        self.polygons.append(polygon)
    def compute_fitness(self):
        self.error = 0
        for poly in self.polygons:
            corner = poly.corner
            length = poly.length
            for x in range(corner[0],min(corner[0]+length,w)):
                for y in range(corner[1],min(corner[1]+length,h)):
                    color = data[y*w+x]
                    self.error += compute(color,poly.color)

    def fitness(self):
        x =  1-self.error/diff
        return x
    def draw(self):
        image = Image.new('RGB', (self.w, self.h), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        for poly in self.polygons:
            corner = poly.corner
            length = poly.length
            for x in range(corner[0],min(corner[0]+length,self.w)):
                for y in range(corner[1],min(corner[1]+length,self.h)):
                    draw.point((x, y), fill=tuple(poly.color))
        image.show()
        image.save('cmp2.png','png')
class Pictures:
    global w,h,img
    def __init__(self,max_iter,capcity,mutation_rate,cross_rate):
        self.max_iter = max_iter
        self.w = w
        self.h = h
        self.size = capcity
        self.chrom_size = 0
        self.mutation_rate = mutation_rate
        self.cross_rate = cross_rate
        self.current_generation = []
        self.next_generation = []
        self.slector = []
        self.base = 2
        #generate pictures and polygons
        first = True
        for i in range(self.size):
            picture = Picture(self.w,self.h)
            for x in range(0,w,self.base):
                for y in range(0,h,self.base):
                    polygon = Polygon(self.base,self.base,self.w,self.h,(x,y))
                    picture.add(polygon)
                    if first:
                        self.chrom_size+=1
            first = False
            #print(len(self.Index))
            self.current_generation.append(picture)
    def select(self):
        t = random.random()
        i = next(i for i,p in enumerate(self.selector) if t<p)
        return i
    def cross(self,pic1,pic2):
        p = random.random()
        if pic1!=pic2 and p<self.cross_rate:
            #swap polygon(chrom)
            pic1,pic2 = copy.deepcopy(pic1),copy.deepcopy(pic2)
            pos = random.randint(0,self.chrom_size-1)
            #swap
            tmp1 = pic1.polygons[pos:]
            tmp2 = pic2.polygons[pos:]
            pic1.polygons = pic1.polygons[:pos]+tmp1
            pic2.polygons = pic2.polygons[:pos]+tmp2
            return pic1,pic2
        return None,None
    def mutate(self,pic):
        p = random.random()
        if p<self.mutation_rate:
            pos = random.randint(0,self.chrom_size-1)
            for poly in pic.polygons[pos:min(pos+3,self.chrom_size-1)]:
                poly.color = [random.randint(0,255) for i in range(3)]
        return pic

    def evaluate(self):
        self.selector = []
        org_prob = 0
        base = 2
        for pic in self.current_generation:
            pic.compute_fitness()
        total = sum(base**(p.fitness()/10) for p in self.current_generation)
        for each in self.current_generation:
            org_prob += base**(each.fitness()/10)
            self.selector.append(org_prob/total)
    def evolution(self):
        self.evaluate()
        num=0
        while num< (self.size//2):
            pos1,pos2 = self.select(),self.select()
            pic1,pic2 = self.current_generation[pos1],self.current_generation[pos2]
            pic1,pic2 =self.cross(pic1,pic2)
            if pic1==None:
                continue
            #mutation
            pic1=self.mutate(pic1)
            pic2=self.mutate(pic2)
            self.next_generation.append(pic1)
            self.next_generation.append(pic2)
            num+=2
        self.current_generation = self.next_generation
        self.next_generation = []

    def run(self):
        self.evaluate()
        for i in range(self.max_iter):
            fit,i1,best = max((pic.fitness(),i,pic) for i,pic in enumerate(self.current_generation))
            self.evolution()
            fitw,i2,worst = min((pic.fitness(),i,pic) for i,pic in enumerate(self.current_generation))
            self.current_generation[i2] = best
            print(i,'best',fit,'worst',fitw)
            if i%50== 1:
                best.draw()