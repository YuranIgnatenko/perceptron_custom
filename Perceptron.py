import numpy as np


# #  ПЕРЦЕПТРОН
class Network():
    def __init__(self, array_data_train, array_data_predict, flag_upload_synaptics, flag_seed=True, num_seed=1, name_net='Net1'):
        self.__name_net = name_net
        self.__data_train = np.array(array_data_train)
        self.__data_predict = np.array(array_data_predict).T
        self.__isfile = False
        self.__count_learn = 0
        self.count_inputs = len(self.__data_train[0])
        np.random.seed(1)
        try:
            if flag_upload_synaptics:
                synaptics_from_file = open('Synaptics.txt', 'r').read()
                # print('syn: ', synaptics_from_file)
                if synaptics_from_file == '':
                    self._synaptic_weights = 2 * np.random.random((len(self.__data_train[0]), 1)) - 1
                else:
                    l = synaptics_from_file.split(' ')
                    syns = []
                    for s in l:
                        if s != '':
                            syns.append(float(s))
                    self._synaptic_weights = np.array([])

                    for s in syns:
                        self._synaptic_weights = np.append(self._synaptic_weights, float(s))
                    # print(self._synaptic_weights)
                    self.__isfile = True
                    print('syn w:  \n', self._synaptic_weights, type(self._synaptic_weights))

            else:
                self._synaptic_weights = 2 * np.random.random((len(self.__data_train[0]), 1)) - 1
        except Exception as e:
            self._synaptic_weights = 2 * np.random.random((len(self.__data_train[0]), 1)) - 1
            # print('syn w except:  \n', self._synaptic_weights, type(self._synaptic_weights), e)
        # print('syn w:  \n',self._synaptic_weights)

    def learn(self,  func_other=None, flag_save=True, count_iter = 20000):
        # МЕТОД ОБРАТНОГО РАСПРОСТРАНЕНИЯ
        self.__count_learn += 1
        training_inputs = self.__data_train
        for i in range(count_iter):
            # МЕТОД ОБРАТНОГО РАСПРОСТРАНЕНИЯ
            input_layer = training_inputs
            outputs = self.sigmoid(np.dot(input_layer, np.array(self._synaptic_weights)))
            err = self.__data_predict - outputs
            adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))
            # print('syn w learn:  \n', self._synaptic_weights, type(self._synaptic_weights))
            self._synaptic_weights = np.array(self._synaptic_weights) + adjustments
            if func_other != None:
                func_other()
        if flag_save:
            f = open('Synaptics.txt', 'w')

            for syns in self._synaptic_weights:
                # print(syns)
                f.write(str(syns[0]) + ' ')

    def learn_if_up(self,  func_other=None, flag_save=True, num_ideal = -90):
        # МЕТОД ОБРАТНОГО РАСПРОСТРАНЕНИЯ
        self.__count_learn += 1
        training_inputs = self.__data_train
        outputs = -99
        while outputs > num_ideal:
            # МЕТОД ОБРАТНОГО РАСПРОСТРАНЕНИЯ
            print(outputs)
            input_layer = training_inputs
            outputs = self.sigmoid(np.dot(input_layer, np.array(self._synaptic_weights)))
            err = self.__data_predict - outputs
            adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))
            # print('syn w learn:  \n', self._synaptic_weights, type(self._synaptic_weights))
            self._synaptic_weights = np.array(self._synaptic_weights) + adjustments
            if func_other is not None:
                func_other()
        if flag_save:
            f = open('Synaptics.txt', 'w')

            for syns in self._synaptic_weights:
                # print(syns)
                f.write(str(syns[0]) + ' ')

    def learn_other(self, data_train, data_pred, func_other=None, flag_save=True, COUNT_ITER=20000):
        self.__count_learn += 1
        # МЕТОД ОБРАТНОГО РАСПРОСТРАНЕНИЯ
        for i in range(COUNT_ITER):
            input_layer = np.array(data_train)
            data_pred = np.array(data_pred).T
            outputs = self.sigmoid(np.dot(input_layer, self._synaptic_weights))

            err = data_pred - outputs
            adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))
            try:
                self._synaptic_weights = np.array(self._synaptic_weights) + adjustments
                print(self._synaptic_weights, adjustments)
            except ValueError:
                print('ERROR !  CHECK IN INIT PARAMETERS NETWORK bool -> (True / False)')
            if func_other != None:
                func_other()
        if flag_save:
            f = open('Synaptics.txt', 'w')
            for syns in self._synaptic_weights:
                # print(syns)
                f.write(str(syns[0]) + ' ')

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_result(self, new_inputs):
        output = self.sigmoid(np.dot(new_inputs, self._synaptic_weights))
        # print(output, self._synaptic_weights)
        return output

    def get_synaptics(self):
        print(self._synaptic_weights)
        return self._synaptic_weights

    def get_bool_file(self):
        return self.__isfile

    def get_learn_count(self):
        return self.__count_learn

    def get_name_net(self):
        return self.__name_net

    def get_count_inputs(self):
        return self.count_inputs



class MoreOutEquallyInputNetwork():
    def __init__(self, count_input_net, array_data_train, array_data_predicts, flag_upload_synaptics=False,
                 flag_seed=False,
                 num_seed=False):
        self.array_inputs_net = []
        self.ar_res = []
        counter_pred_ar = 0

        counter_elem_pred = 0
        counter_ar_train = 0
        for i in range(count_input_net):
            print(array_data_train[counter_ar_train])
            net = Network([array_data_train[counter_ar_train]], array_data_predicts[counter_pred_ar][counter_elem_pred], flag_upload_synaptics, flag_seed,
                          num_seed,
                          name_net=f'Net_{i}')
            counter_elem_pred += 1
            self.array_inputs_net.append(net)
        print(self.array_inputs_net, len(self.array_inputs_net))


    # def learn_all(self, count_iter=10000, func_other=None, flag_save=False):
    #
    #     counter_for_predict_from_array = -1
    #
    #     for net in self.array_inputs_net:
    #         print(net)
    #         print("\n" * 20 + str(counter_for_predict_from_array))
    #         counter_for_predict_from_array += 1
    #         # МЕТОД ОБРАТНОГО РАСПРОСТРАНЕНИЯ
    #         net.count_learn += 1
    #         training_inputs = net.data_train
    #         for i in range(count_iter):
    #             # МЕТОД ОБРАТНОГО РАСПРОСТРАНЕНИЯ
    #             input_layer = training_inputs
    #             outputs = net.sigmoid(np.dot(input_layer, np.array(net.synaptic_weights)))
    #             err = net.data_predict[counter_for_predict_from_array] - outputs
    #             adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))
    #             # print('syn w learn:  \n', self.synaptic_weights, type(self.synaptic_weights))
    #             net.synaptic_weights = np.array(net.synaptic_weights) + adjustments
    #             if func_other != None:
    #                 func_other()
    #         if flag_save:
    #             f = open('Synaptics.txt', 'w')
    #
    #             for syns in net.synaptic_weights:
    #                 # print(syns)
    #                 f.write(str(syns[0]) + ' ')

    def learner(self, data_train, data_pred, count_iter=20000, func_other=None, flag_save=False):
        counter_pred_ar = 0
        for ar in data_train:
            counter_elem_pred = 0
            for net in self.array_inputs_net:
                # print(f"INDEX >>>  {counter_pred_ar, counter_elem_pred, ar,data_pred}")
                # print(ar, data_pred[counter_pred_ar][counter_elem_pred], net)
                net.learn_other([ar], data_pred[counter_pred_ar][counter_elem_pred], func_other, flag_save, COUNT_ITER=count_iter)
                counter_elem_pred += 1
            counter_pred_ar += 1

    def get_result(self, input_data):
        for net in self.array_inputs_net:
            # print(net)
            res = net.get_result(input_data)
            self.ar_res.append(res[0])
        # print(f"ar res {self.ar_res}")
        return self.ar_res




class MoreOutNotEquallyInputNetwork():
    def __init__(self, count_input_net, count_output_net,array_data_train, array_data_predicts, flag_upload_synaptics=False,
                 flag_seed=False,
                 num_seed=False):
        self.array_outputs_net = count_output_net
        self.array_inputs_net = []
        self.ar_res = []
        counter_pred_ar = 0

        counter_elem_pred = 0
        counter_ar_train = 0
        for i in range(count_output_net):
            # print(array_data_train[counter_ar_train])
            net = Network([array_data_train[counter_ar_train]], array_data_predicts[counter_pred_ar][counter_elem_pred], flag_upload_synaptics, flag_seed,
                          num_seed,
                          name_net=f'Net_{i}')
            counter_elem_pred += 1
            self.array_inputs_net.append(net)
        # print(self.array_inputs_net, len(self.array_inputs_net))


    # def learn_all(self, count_iter=10000, func_other=None, flag_save=False):
    #
    #     counter_for_predict_from_array = -1
    #
    #     for net in self.array_inputs_net:
    #         print(net)
    #         print("\n" * 20 + str(counter_for_predict_from_array))
    #         counter_for_predict_from_array += 1
    #         # МЕТОД ОБРАТНОГО РАСПРОСТРАНЕНИЯ
    #         net.count_learn += 1
    #         training_inputs = net.data_train
    #         for i in range(count_iter):
    #             # МЕТОД ОБРАТНОГО РАСПРОСТРАНЕНИЯ
    #             input_layer = training_inputs
    #             outputs = net.sigmoid(np.dot(input_layer, np.array(net.synaptic_weights)))
    #             err = net.data_predict[counter_for_predict_from_array] - outputs
    #             adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))
    #             # print('syn w learn:  \n', self.synaptic_weights, type(self.synaptic_weights))
    #             net.synaptic_weights = np.array(net.synaptic_weights) + adjustments
    #             if func_other != None:
    #                 func_other()
    #         if flag_save:
    #             f = open('Synaptics.txt', 'w')
    #
    #             for syns in net.synaptic_weights:
    #                 # print(syns)
    #                 f.write(str(syns[0]) + ' ')

    def learner(self, data_train, data_pred, count_iter=20000, func_other=None, flag_save=False):
        counter_pred_ar = 0
        for ar in data_train:
            counter_elem_pred = 0
            for net in self.array_inputs_net:
                # print(f"INDEX >>>  {counter_pred_ar, counter_elem_pred, ar,data_pred}")
                # print(ar, data_pred[counter_pred_ar][counter_elem_pred], net)
                net.learn_other([ar], data_pred[counter_pred_ar][counter_elem_pred], func_other, flag_save, COUNT_ITER=count_iter)
                counter_elem_pred += 1
            counter_pred_ar += 1

    def get_result(self, input_data):
        for net in self.array_inputs_net:
            # print(net)
            res = net.get_result(input_data)
            self.ar_res.append(res[0])
        print(f"ar res {self.ar_res}")
        return self.ar_res




# class MoreOutNet2():
#     def __init__(self, count_input_net, array_data_train, array_data_predicts, flag_upload_synaptics=False,
#                      flag_seed=True,
#                      num_seed=1):
#         self.array_inputs_net = []
#         self.ar_res = []
#         counter_pred_ar = 0
#         for i in range(count_input_net):
#             net = Network(array_data_train[0], array_data_predicts[0][counter_pred_ar], flag_upload_synaptics, flag_seed,
#                           num_seed,
#                           name_net=f'Net_{i}')
#             counter_pred_ar += 1
#             self.array_inputs_net.append(net)
#             print(self.array_inputs_net)


if __name__ == '__main__':
    training_inputs = [[180, 80],
                       [190, 90],
                       [170, 70],
                       [160, 60],
                       [150, 50],
                       [180, 60],
                       [190, 70],
                       [170, 50],
                       [160, 40],
                       [150, 30]]


    training_outputs = [[1, 0, 1],
                        [1, 0, 1],
                        [1, 0, 1],
                        [1, 0, 1],
                        [1, 0, 1],
                        [0, 1, 1],
                        [0, 1, 1],
                        [0, 1, 1],
                        [0, 1, 1],
                        [0, 1, 1]
                        ]

    nets = MoreOutNotEquallyInputNetwork(2, 3, training_inputs, training_outputs)
    nets.learner(training_inputs, training_outputs, count_iter=2000)
    nets.get_result(np.array([180,80]))

    print('\n' + "+" * 20 + '\n')

    # training_inputs = [[0, 0, 1],
    #                    [1, 1, 1],
    #                    [1, 0, 1],
    #                    [0, 1, 1]]
    #
    # training_outputs = [[0, 1, 1, 0]]
    #
    # net = Network(training_inputs, training_outputs, False)
    # net.learn()
    # # net.learn_if_up(num_ideal=0.9)
    # res = net.get_result(np.array([1, 1, 0]))
    # print(res)  # [0.99996185]

    # tr2 = [[0.0],
    #        [0.9]]
    # pr2 = [[0, 1]]
    # net2 = Network(tr2, pr2, flag_upload_synaptics=False)
    #
    # net2.learn(count_iter=100000)
    #
    # res2 = net2.get_result(np.array([0.3]))
    #
    # print(res2)

    # training_inputs = [[1, 1, 1],[1, 1, 1],[1, 1, 1],[1, 1, 1]]
    # training_outputs = [[1,1,1,1]]
    # net = Network(training_inputs, training_outputs, False)
    # net.learn_other([[0, 0, 1]],[[0]])
    # net.learn_other([[1, 1, 1]],[[1]])
    # net.learn_other([[1, 0, 1]],[[1]])
    # net.learn_other([[0, 1, 1]],[[0]])
    #
    # res = net.get_result(np.array([1, 1, 0]))
    # print(res) #[0.98854986]
