"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/shared/api/axios";
import { useState } from "react";
import { Modal, Input, Button, Form, Spin, Pagination } from "antd";

import { useCreateTodo } from "@/entites/todo/api/useCreateTodo";
import { useDeleteTodo } from "@/entites/todo/api/useDeleteTodo";
import { useUpdateTodo } from "@/entites/todo/api/useUpdateTodo";
import { useToggleFavorite } from "@/entites/todo/api/useToggleFavorite";
import { useTodoStore } from "@/entites/todo/api/useTodoStore";

export default function HomePage() {
  const { page, pageSize, setPage, setPageSize } = useTodoStore();

 
  const { data, isLoading } = useQuery({
    queryKey: ["todos", page, pageSize],
    queryFn: async () => {
      const res = await api.get(
        `/todos?_page=${page}&_limit=${pageSize}`
      );
      return res.data;
    },
  });

  const { mutate: createTodo } = useCreateTodo();
  const { mutate: deleteTodo } = useDeleteTodo();
  const { mutate: updateTodo } = useUpdateTodo();
  const { mutate: toggleFavorite } = useToggleFavorite();
  //  мутации создание удаление редактрование 

  const [createOpen, setCreateOpen] = useState(false); 
  // открывает create todo 
  const [editOpen, setEditOpen] = useState(false);
  // открывает эдит
  const [selectedTodo, setSelectedTodo] = useState<any>(null);
  // обект редоактиbруемого todo 

  const [createForm] = Form.useForm();
  const [editForm] = Form.useForm();

  const handleCreate = (values: any) => {
    createTodo(values);
    setCreateOpen(false);
    createForm.resetFields();
  };

  const openEditModal = (todo: any) => {
    setSelectedTodo(todo);
    editForm.setFieldsValue(todo);
    setEditOpen(true);
  };

  const handleUpdate = (values: any) => {
    updateTodo({
      id: selectedTodo.id,
      data: values,
    });

    setEditOpen(false);
    setSelectedTodo(null);
  };

  if (isLoading) {
    return (
      <div style={{ padding: 20 }}>
        <Spin />
      </div>
    );
  }

  return (
    <div style={{ padding: 20 }}>
      <h1>Todo List</h1>

      <Button type="primary" onClick={() => setCreateOpen(true)}>
        + Add Todo
      </Button>

      {data?.map((todo: any) => (
        <div
          key={todo.id}
          style={{
            marginTop: 10,
            padding: 10,
            border: "1px solid #ddd",
            borderRadius: 6,
          }}
        >
          <h3>{todo.title}</h3>
          <p>{todo.description}</p>

          <div style={{ display: "flex", gap: 8 }}>
            <Button onClick={() => openEditModal(todo)}>
              Edit
            </Button>

            <Button danger onClick={() => deleteTodo(todo.id)}>
              Delete
            </Button>

            <Button
              onClick={() =>
                toggleFavorite({
                  id: todo.id,
                  favorite: !todo.favorite,
                })
              }
            >
              {todo.favorite ? "⭐" : "☆"}
            </Button>
          </div>
        </div>
      ))}

      <div style={{ marginTop: 20 }}>
        <Pagination
          current={page}
          pageSize={pageSize}
          onChange={(p, size) => {
            setPage(p);
            setPageSize(size);
          }}
          showSizeChanger
        />
      </div>

      <Modal
        title="Create Todo"
        open={createOpen}
        onCancel={() => setCreateOpen(false)}
        footer={null}
      >
        <Form form={createForm} onFinish={handleCreate}>
          <Form.Item name="title" rules={[{ required: true }]}>
            <Input placeholder="Title" />
          </Form.Item>

          <Form.Item name="description" rules={[{ required: true }]}>
            <Input.TextArea placeholder="Description" />
          </Form.Item>

          <Button type="primary" htmlType="submit" block>
            Create
          </Button>
        </Form>
      </Modal>

      <Modal
        title="Edit Todo"
        open={editOpen}
        onCancel={() => setEditOpen(false)}
        footer={null}
      >
        <Form form={editForm} onFinish={handleUpdate}>
          <Form.Item name="title" rules={[{ required: true }]}>
            <Input placeholder="Title" />
          </Form.Item>

          <Form.Item name="description" rules={[{ required: true }]}>
            <Input.TextArea placeholder="Description" />
          </Form.Item>

          <Button type="primary" htmlType="submit" block>
            Update
          </Button>
        </Form>
      </Modal>
    </div>
  );
}