# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random as rnd
import sys

import numpy as np


class CGP:
    class CGPFunc:
        def __init__(self, f, name, arity):
            self.function = f
            self.name = name
            self.arity = arity

    class CGPNode:
        def __init__(self, args, f):
            self.args = args
            self.function = f

    def __init__(
        self,
        genome,
        num_inputs,
        num_outputs,
        num_cols,
        num_rows,
        library,
        recurrency_distance=1.0,
    ):
        self.genome = genome.copy()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.max_graph_length = num_cols * num_rows
        self.library = library
        self.max_arity = 0
        self.recurrency_distance = recurrency_distance
        for f in self.library:
            self.max_arity = np.maximum(self.max_arity, f.arity)
        self.graph_created = False

    def create_graph(self):
        self.to_evaluate = np.zeros(self.max_graph_length, dtype=bool)
        self.node_output = np.zeros(
            self.max_graph_length + self.num_inputs, dtype=np.float64
        )
        self.nodes_used = []
        self.output_genes = np.zeros(self.num_outputs, dtype=np.int_)
        self.nodes = np.empty(0, dtype=self.CGPNode)
        for i in range(0, self.num_outputs):
            self.output_genes[i] = self.genome[len(self.genome) - self.num_outputs + i]
        i = 0
        # building node list
        while i < len(self.genome) - self.num_outputs:
            f = self.genome[i]
            args = np.empty(0, dtype=int)
            for j in range(self.max_arity):
                args = np.append(args, self.genome[i + j + 1])
            i += self.max_arity + 1
            self.nodes = np.append(self.nodes, self.CGPNode(args, f))
        self.node_to_evaluate()
        self.graph_created = True

    def node_to_evaluate(self):
        p = 0
        while p < self.num_outputs:
            if self.output_genes[p] - self.num_inputs >= 0:
                self.to_evaluate[self.output_genes[p] - self.num_inputs] = True
            p = p + 1
        p = self.max_graph_length - 1
        while p >= 0:
            if self.to_evaluate[p]:
                for i in range(0, len(self.nodes[p].args)):
                    arg = self.nodes[p].args[i]
                    if arg - self.num_inputs >= 0:
                        self.to_evaluate[arg - self.num_inputs] = True
                self.nodes_used.append(p)
            p = p - 1
        self.nodes_used = np.array(self.nodes_used)

    def load_input_data(self, input_data):
        for p in range(self.num_inputs):
            self.node_output[p] = input_data[p]

    def compute_graph(self):
        self.node_output_old = self.node_output.copy()
        p = len(self.nodes_used) - 1
        while p >= 0:
            args = np.zeros(self.max_arity)
            for i in range(0, self.max_arity):
                args[i] = self.node_output_old[self.nodes[self.nodes_used[p]].args[i]]
            f = self.library[self.nodes[self.nodes_used[p]].function].function
            self.node_output[self.nodes_used[p] + self.num_inputs] = f(args)

            if (
                self.node_output[self.nodes_used[p] + self.num_inputs]
                != self.node_output[self.nodes_used[p] + self.num_inputs]
            ):
                print(
                    self.library[self.nodes[self.nodes_used[p]].function].name,
                    " returned NaN with ",
                    args,
                )
            if (
                self.node_output[self.nodes_used[p] + self.num_inputs] < -1.0
                or self.node_output[self.nodes_used[p] + self.num_inputs] > 1.0
            ):
                print(
                    self.library[self.nodes[self.nodes_used[p]].function].name,
                    " returned ",
                    self.node_output[self.nodes_used[p] + self.num_inputs],
                    " with ",
                    args,
                )

            p = p - 1

    def run(self, inputData):
        if not self.graph_created:
            self.create_graph()

        self.load_input_data(inputData)
        self.compute_graph()
        return self.read_output()

    def read_output(self):
        output = np.zeros(self.num_outputs)
        for p in range(0, self.num_outputs):
            output[p] = self.node_output[self.output_genes[p]]
        return output

    def clone(self):
        return CGP(
            self.genome,
            self.num_inputs,
            self.num_outputs,
            self.num_cols,
            self.num_rows,
            self.library,
        )

    def mutate(self, num_mutationss):
        for i in range(0, num_mutationss):
            index = rnd.randint(0, len(self.genome) - 1)
            if index < self.num_cols * self.num_rows * (self.max_arity + 1):
                # this is an internal node
                if index % (self.max_arity + 1) == 0:
                    # mutate function
                    self.genome[index] = rnd.randint(0, len(self.library) - 1)
                else:
                    # mutate connection
                    self.genome[index] = rnd.randint(
                        0,
                        self.num_inputs
                        + (int(index / (self.max_arity + 1)) - 1) * self.num_rows,
                    )
            else:
                # this is an output node
                self.genome[index] = rnd.randint(
                    0, self.num_inputs + self.num_cols * self.num_rows - 1
                )

    def mutate_per_gene(self, mutation_rate_nodes, mutation_rate_outputs):
        for index in range(0, len(self.genome)):
            if index < self.num_cols * self.num_rows * (self.max_arity + 1):
                # this is an internal node
                if rnd.random() < mutation_rate_nodes:
                    if index % (self.max_arity + 1) == 0:
                        # mutate function
                        self.genome[index] = rnd.randint(0, len(self.library) - 1)
                    else:
                        # mutate connection
                        self.genome[index] = rnd.randint(
                            0,
                            min(
                                self.max_graph_length + self.num_inputs - 1,
                                (
                                    self.num_inputs
                                    + (int(index / (self.max_arity + 1)) - 1)
                                    * self.num_rows
                                )
                                * self.recurrency_distance,
                            ),
                        )
                        # self.genome[index] = rnd.randint(0, self.num_inputs + (int(index / (self.max_arity + 1)) - 1) * self.num_rows)
            else:
                # this is an output node
                if rnd.random() < mutation_rate_outputs:
                    # this is an output node
                    self.genome[index] = rnd.randint(
                        0, self.num_inputs + self.num_cols * self.num_rows - 1
                    )

    def to_dot(self, file_name, input_names, output_names):
        if not self.graph_created:
            self.create_graph()
        out = open(file_name, "w")
        out.write("digraph cgp {\n")
        out.write('\tsize = "4,4";\n')
        self.dot_rec_visited_nodes = np.empty(1)
        for i in range(self.num_outputs):
            out.write("\t" + output_names[i] + " [shape=oval];\n")
            self._write_dot_from_gene(
                output_names[i], self.output_genes[i], out, 0, input_names, output_names
            )
        out.write("}")
        out.close()

    def _write_dot_from_gene(self, to_name, pos, out, a, input_names, output_names):
        if pos < self.num_inputs:
            out.write("\t" + input_names[pos] + " [shape=polygon,sides=6];\n")
            out.write(
                "\t"
                + input_names[pos]
                + " -> "
                + to_name
                + ' [label="'
                + str(a)
                + '"];\n'
            )
            self.dot_rec_visited_nodes = np.append(self.dot_rec_visited_nodes, [pos])
        else:
            pos -= self.num_inputs
            out.write(
                "\t"
                + self.library[self.nodes[pos].function].name
                + "_"
                + str(pos)
                + " -> "
                + to_name
                + ' [label="'
                + str(a)
                + '"];\n'
            )
            if pos + self.num_inputs not in self.dot_rec_visited_nodes:
                out.write(
                    "\t"
                    + self.library[self.nodes[pos].function].name
                    + "_"
                    + str(pos)
                    + " [shape=none];\n"
                )
                for a in range(self.library[self.nodes[pos].function].arity):
                    self._write_dot_from_gene(
                        self.library[self.nodes[pos].function].name + "_" + str(pos),
                        self.nodes[pos].args[a],
                        out,
                        a,
                        input_names,
                        output_names,
                    )
            self.dot_rec_visited_nodes = np.append(
                self.dot_rec_visited_nodes, [pos + self.num_inputs]
            )

    def to_function_string(self, input_names, output_names):
        if not self.graph_created:
            self.create_graph()
        for o in range(self.num_outputs):
            print(output_names[o] + " = ", end="")
            self._write_from_gene(self.output_genes[o], input_names, output_names)
            print(";")
            print("")

    def _write_from_gene(self, pos, input_names, output_names):
        if pos < self.num_inputs:
            print(input_names[pos], end="")
        else:
            pos -= self.num_inputs
            print(self.library[self.nodes[pos].function].name + "(", end="")
            for a in range(self.library[self.nodes[pos].function].arity):
                # print(' ', end='')
                self._write_from_gene(
                    self.nodes[pos].args[a], input_names, output_names
                )
                if a != self.library[self.nodes[pos].function].arity - 1:
                    print(", ", end="")
                # else:
                # 	print(')', end='')
            print(")", end="")

    @classmethod
    def random(
        cls, num_inputs, num_outputs, num_cols, num_rows, library, recurrency_distance
    ):
        max_arity = 0
        for f in library:
            max_arity = np.maximum(max_arity, f.arity)
        genome = np.zeros(
            num_cols * num_rows * (max_arity + 1) + num_outputs, dtype=int
        )
        gPos = 0
        for c in range(0, num_cols):
            for r in range(0, num_rows):
                genome[gPos] = rnd.randint(0, len(library) - 1)
                for a in range(max_arity):
                    genome[gPos + a + 1] = rnd.randint(0, num_inputs + c * num_rows - 1)
                gPos = gPos + max_arity + 1
        for o in range(0, num_outputs):
            genome[gPos] = rnd.randint(0, num_inputs + num_cols * num_rows - 1)
            gPos = gPos + 1
        return CGP(
            genome,
            num_inputs,
            num_outputs,
            num_cols,
            num_rows,
            library,
            recurrency_distance,
        )

    def save(self, file_name):
        out = open(file_name, "w")
        out.write(str(self.num_inputs) + " ")
        out.write(str(self.num_outputs) + " ")
        out.write(str(self.num_cols) + " ")
        out.write(str(self.num_rows) + "\n")
        for g in self.genome:
            out.write(str(g) + " ")
        out.write("\n")
        for f in self.library:
            out.write(f.name + " ")
        out.close()

    @classmethod
    def load_from_file(cls, file_name, library):
        inp = open(file_name, "r")
        pams = inp.readline().split()
        genes = inp.readline().split()
        funcs = inp.readline().split()
        inp.close()
        params = np.empty(0, dtype=int)
        for p in pams:
            params = np.append(params, int(p))
        genome = np.empty(0, dtype=int)
        for g in genes:
            genome = np.append(genome, int(g))
        return CGP(genome, params[0], params[1], params[2], params[3], library)

    @classmethod
    def test(cls, num):
        c = CGP.random(2, 1, 2, 2, 2)
        for i in range(0, num):
            c.mutate(1)
            print(c.genome)
            print(c.run([1, 2]))
